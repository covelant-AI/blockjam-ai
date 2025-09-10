from ultralytics import YOLO

def build_engines(player_model_path, ball_model_path):
    print(10 * "#" + "Exporting Player Engine" + 10 * "#")
    model = YOLO(player_model_path)
    player_model_path = model.export(format="engine", imgsz=640, half=True, batch=1)

    print(10 * "#" + "Exporting Balls Engine" + 10 * "#")
    model = YOLO(ball_model_path)
    ball_model_path = model.export(format="engine", imgsz=(736, 1280), half=True, batch=1)
    print("Finished Compiling Engines")

    return player_model_path, ball_model_path