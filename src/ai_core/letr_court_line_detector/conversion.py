def reorder_court_keypoints(court_keypoints: list[float]) -> list[float]:
    """
    Convert court keypoints from LETR to proper order
    """
    first_config = [
        0,3,8,11, # Outer Corners
        1,9,2,10, # Inner Corners
        4,5,6,7, # Service Box Corners
        12,13 # Service Box Center
        ]


    second_config = [
        8,11,0,3, # Outer Corners
        9,1,10,2, # Inner Corners
        6,7,4,5, # Service Box Corners
        13,12 # Service Box Center
        ]

    config = first_config
    if court_keypoints[0][1] > court_keypoints[11][1]:
        config = second_config

    return [court_keypoints[i] for i in config]
    


    