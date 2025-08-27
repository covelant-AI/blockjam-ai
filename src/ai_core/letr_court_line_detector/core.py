from .letr_inference import LETRInference
from .lines import tennis_court_model_points, tennis_court_model_lines
import numpy as np
import cv2
import sys
import networkx as nx
from shapely.geometry import LineString
from .conversion import reorder_court_keypoints
from utils.conversions import scale_points_to_size

class LETRCourtDetector:
    def __init__(self, model_path, device):
        self.model = LETRInference(model_path, 0.25)

    
    def linesFiltering(self, lines, imgRes, angleTh = 5, distTh = 10, minLength = 0.1):
        out = []
        minRes = min(imgRes)
        for i, line1 in enumerate(lines):
            append = True
            v1 = line1[2:4] - line1[0:2]
            len1 = np.linalg.norm(v1)
            if len1 < minRes * minLength:
                continue

            for j, line2 in enumerate(lines):
                if i == j:
                    continue
                v2 = line2[2:4] - line2[0:2]
                len2 = np.linalg.norm(v2)

                if len2 < minRes * minLength:
                    continue

                dot = np.dot(v1 / len1, v2 / len2)
                dot = max(-1, min(dot, 1))
                angle = np.arccos(dot) * 180 / np.pi

                angleCondition = np.abs(angle) < angleTh or (angle > 180 - angleTh and angle < 180 + angleTh)

                dist1 = np.linalg.norm(line1[0:2]-line2[0:2]) < distTh
                dist2 = np.linalg.norm(line1[0:2]-line2[2:4]) < distTh
                dist3 = np.linalg.norm(line1[2:4]-line2[0:2]) < distTh
                dist4 = np.linalg.norm(line1[2:4]-line2[2:4]) < distTh
                distCondition = dist1 or dist2 or dist3 or dist4
                # dist1 = pointLineMinDist(line1, line2[0:2])
                # dist2 = pointLineMinDist(line1, line2[2:4])
                # dist3 = pointLineMinDist(line2, line1[0:2])
                # dist4 = pointLineMinDist(line2, line1[2:4])
                # distCondition = min((dist1, dist2, dist3, dist4)) < distTh

                if angleCondition and distCondition and len1 < len2:
                    append = False
                    break

            if append:
                out.append(line1)

        return np.asarray(out)
    
    def linesFilteringWithGraph(self, lines, min_components = 3, lineExtension = 2, hardCut = True):
        def extendLine(line, extension): # (a.x,a.y,b.x,b.y)
            ab = line[2:4] - line[0:2]
            v = (ab / np.linalg.norm(ab)) * extension
            return [line[0:2] - v, line[2:4] + v]
        
        # print(lineExtension)
        G = nx.Graph()
        for i, line1 in enumerate(lines):
            shLine1 = extendLine(line1, lineExtension)
            shLine1 = LineString(shLine1)
            for j, line2 in enumerate(lines[(i+1):]):
                shLine2 = extendLine(line2, lineExtension)
                shLine2 = LineString(shLine2)
                if shLine1.intersects(shLine2):
                    # p = shLine1.intersection(shLine2)
                    G.add_edge(i, i+j+1)
                elif not G.has_node(i):
                    G.add_node(i)
                elif not G.has_node(i+j+1):
                    G.add_node(i+j+1)
        out = np.array([]).reshape(0,4)
        comps = nx.algorithms.components.connected_components(G)
        if hardCut:
            comps = np.array(list(comps))
            sorted = np.array([len(x) for x in comps]).argsort()[::-1]
            comps = comps[sorted][:2]
        for comp in comps:
            if len(comp) >= min_components:
                indices = np.asarray(list(comp))
                out = np.concatenate((out, lines[indices]), axis=0)
        return out
    
    def selectInOrderGenerator(self, size):
        out = [0,0]
        yield np.asarray(out)
        while out[0] != size - 2 or out[1] != size -1:
            out[1] += 1
            if out[1] == size:
                out[0] += 1
                out[1] = out[0]+1
            yield np.array(out)

    def pointLineMinDist(self, line, point): # (a.x,a.y,b.x,b.y), (p.x,p.y)
        ap = point - line[0:2]
        ab = line[2:4] - line[0:2]
        perpendicular_intersection = line[0:2] + max(0, min(1, np.dot(ap, ab)/(ab**2).sum())) * ab
        return np.linalg.norm(perpendicular_intersection - point)

    def computeLineScore(self, projectedLines, lines, angleTh = 4):
        score = 0
        for pLine in projectedLines:
            v1 = pLine[2:4] - pLine[0:2]
            len1 = np.linalg.norm(v1)
            minScore = 1e5
            mini = -1
            for i, line in enumerate(lines):
                localScore = 0
                v2 = line[2:4] - line[0:2]
                len2 = np.linalg.norm(v2)
                dot = abs(np.dot(v1 / len1, v2 / len2))
                dot = min(1, dot)
                angle = np.arccos(dot) * 180 / np.pi
                if angle < angleTh:
                    dist1 = self.pointLineMinDist(pLine, line[0:2])
                    dist2 = self.pointLineMinDist(pLine, line[2:4])
                    dist3 = self.pointLineMinDist(line, pLine[0:2])
                    dist4 = self.pointLineMinDist(line, pLine[2:4])
                    minDist = min((dist1, dist2, dist3, dist4))
                    if minDist < 50:
                        localScore = (angleTh-angle) * 200
                        dist1 = np.sum((pLine[0:2] - line[0:2])**2)
                        dist2 = np.sum((pLine[0:2] - line[2:4])**2)
                        dist3 = np.sum((pLine[2:4] - line[0:2])**2)
                        dist4 = np.sum((pLine[2:4] - line[2:4])**2)
                        localScore += (min(dist1, dist2) + min(dist3, dist4))**2
                        localScore += minDist**2
                        if minScore > localScore:
                            minScore = localScore
                            mini = i
            if mini != -1:
                lines = np.delete(lines, mini, axis=0)
            score += 1e3 - minScore
        return score

    def detect(self, image, max_attempts=10000, processing_resolution=(640, 360), min_score=-450000):
        original_height, original_width = image.shape[:2]
        image = cv2.resize(image, processing_resolution)
        lines = self.model.evaluate(image)
        nlines = len(lines)
        print('image resolution: ', image.shape[:2])
        print('number of lines: ', nlines)

        ### VISUAL DEBUG ###
        # showImgWithLines(image, lines, 'nofilter', True)
        ### END VISUAL DEBUG ###

        # line based filtering
        print('removing lines too close...')
        lines = self.linesFiltering(lines, image.shape[:2])
        print('lines removed: ', nlines - len(lines), "\t remaining: ", len(lines))

        # graph base filtering (a line is connected with another if they intersect)
        nLines = len(lines)
        print('removing lonely lines...')
        lines = self.linesFilteringWithGraph(lines, lineExtension=(min(image.shape[:2])/20))
        print('removed lines: ', nLines - len(lines), "\t remaining: ",len(lines))

        ### VISUAL DEBUG ###
        # showImgWithLines(image, lines, "filter", True)
        ### END VISUAL DEBUG ###

        best_RT_matrix = None
        best_score = float('-inf')
        best_fitting_points = []
        best_projected_points = None

        # create 2 generators to do line selection
        lineGenerator = self.selectInOrderGenerator(lines.shape[0])
        modelLineGenerator = self.selectInOrderGenerator(tennis_court_model_lines.shape[0])

        for i in range(max_attempts):
            # select model and image line pairs
            if  i == 0:
                select_model_lines_idx = next(modelLineGenerator)
            try:
                select_lines_idx = next(lineGenerator)
            except StopIteration:
                lineGenerator = self.selectInOrderGenerator(lines.shape[0])
                select_lines_idx = next(lineGenerator)
                try:
                    select_model_lines_idx = next(modelLineGenerator)
                except StopIteration:
                    break
            
            # select points from the selected lines
            select_points = np.asarray([np.append(lines[select_lines_idx,0], lines[select_lines_idx,2]),np.append(lines[select_lines_idx,1], lines[select_lines_idx,3])]).T
            
            select_model_points = tennis_court_model_points[np.append(tennis_court_model_lines[select_model_lines_idx,0], tennis_court_model_lines[select_model_lines_idx,1])]

            # create matrix from homography of 4 points
            RT_matrix, mask = cv2.findHomography(select_model_points.astype(np.float32)[:, np.newaxis, :], select_points.astype(np.float32)[:, np.newaxis, :])
            
            if RT_matrix is None or np.sum(np.isinf(RT_matrix)) != 0:
                # print("RT matrix contains an infinite")
                continue
            
            if abs(np.linalg.det(RT_matrix)) < sys.float_info.epsilon:
                # print("Determinant equal to zero")
                continue

            # reproject points from the model
            tennis_court_projected_points = RT_matrix @ np.r_[tennis_court_model_points.T, np.full((1, tennis_court_model_points.shape[0]), 1, dtype=np.float32)]
            if 0 in tennis_court_projected_points[2]:
                continue
            tennis_court_projected_points = tennis_court_projected_points / tennis_court_projected_points[2]
            tennis_court_projected_points = tennis_court_projected_points.T
            
            # reproject lines from the model
            projected_lines = []
            for line in tennis_court_model_lines:
                projected_lines.append(np.append(tennis_court_projected_points[line[0]][0:2], tennis_court_projected_points[line[1]][0:2]))
            projected_lines  = np.asarray(projected_lines)

            # compute score
            score = self.computeLineScore(projected_lines, lines)
            
            # compare with the best
            if best_score < score:
                best_score = score
                best_RT_matrix = RT_matrix
                best_fitting_points = select_points
                best_projected_points = tennis_court_projected_points
            
            if i% 50 == 0:
                print("\rfitting attempts: ",i,"  best score: ", best_score, end='')
        
        best_fitting_points = np.asarray(best_fitting_points)

        print("\nbest_score:", best_score)
        if best_score < min_score:
            raise Exception("No good fit found")

        # produce image with projected homography

        best_projected_points = reorder_court_keypoints(best_projected_points)
        best_projected_points = scale_points_to_size(best_projected_points, processing_resolution, (original_width, original_height))
        return best_projected_points