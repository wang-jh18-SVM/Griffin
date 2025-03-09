import os
import argparse
import cv2
import numpy as np
from data_utils import load_img, load_calibration, load_pose, load_label
from space_utils import (
    proj_ego_to_sensor,
    proj_cam_to_img,
    proj_ego_to_ENU,
    proj_object_to_ego,
)
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class LabelVisualizer:
    def __init__(self, data_base):
        self.data_base = data_base
        self.label_dir = os.path.join(data_base, "label")
        self.frame_list = [label.split(".")[0] for label in os.listdir(self.label_dir)]
        self.frame_list.sort()

        self.output_path = os.path.join(data_base, "label_visualization")
        if "vehicle-side" in data_base:
            self.sensors = ["front", "back", "left", "right"]  # For vehicle
        else:
            self.sensors = ["bottom", "front", "back", "left", "right"]  # For drone
        self.sensor_calib = {}
        os.makedirs(self.output_path, exist_ok=True)
        for sensor in self.sensors:
            os.makedirs(os.path.join(self.output_path, sensor), exist_ok=True)
            self.sensor_calib[sensor] = load_calibration(data_base, sensor)
        # os.makedirs(os.path.join(self.output_path, "utm"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "ego"), exist_ok=True)

        # Define colors for different object types: Military, Soldier, Pedestrian, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle
        self.colors = {
            "rider": (255, 0, 255),
            "car": (255, 255, 0),
            "truck": (255, 255, 0),
            "bus": (255, 255, 0),
            "train": (255, 255, 0),
            "motorcycle": (255, 0, 255),
            "bicycle": (255, 0, 255),
            "default": (0, 255, 255),
        }

    def get_3d_box_corners(self, obj):
        """
        Get the 8 corners of a 3D bounding box given the object data.
        """
        # Half-dimensions
        dx = obj.l / 2
        dy = obj.w / 2
        dz = obj.h / 2

        # Corners of the bounding box in the object's local frame (BEV coordinates)
        corners = np.array(
            [
                [dx, dy, dz],
                [dx, -dy, dz],
                [-dx, -dy, dz],
                [-dx, dy, dz],  # Top four corners
                [dx, dy, -dz],
                [dx, -dy, -dz],
                [-dx, -dy, -dz],
                [-dx, dy, -dz],  # Bottom four corners
            ]
        )

        # # Rotation matrix around the Z axis (yaw)
        # yaw = obj.yaw
        # rotation_matrix = np.array(
        #     [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        # )

        # # Rotate and translate the corners
        # rotated_corners = np.dot(corners, rotation_matrix.T)
        # translated_corners = rotated_corners + np.array([obj.x, obj.y, obj.z])

        translated_corners = proj_object_to_ego(corners, obj)

        return translated_corners

    def draw_utm_box(self, image, corners_utm, obj_type, obj_id="0", utm_range=300):
        """
        Draw a 2D bounding box on the utm image.
        Args:
            image (np.ndarray): The image on which to draw.
            corners_utm (np.ndarray): The 2D coordinates of the bounding box corners.
            obj_type (str): The object type (e.g., "People", "Vehicle").
        """
        assert corners_utm.shape == (4, 2)

        # Scale the corners to the image size
        corners_utm = corners_utm * (image.shape[0] / utm_range) + image.shape[0] / 2
        corners_utm[:, 1] = image.shape[0] - corners_utm[:, 1]

        # Define color and thickness
        color = self.colors.get(obj_type.lower(), self.colors['default'])
        thickness = 2

        # Define the edges of the box (connecting the corners)
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Top four edges
        ]

        # Draw the edges
        for edge in edges:
            pt1 = tuple(corners_utm[edge[0]].astype(int))
            pt2 = tuple(corners_utm[edge[1]].astype(int))
            cv2.line(image, pt1, pt2, color, thickness)

        # Add label text (object type) at the top left corner of the box
        label_pos = tuple(corners_utm[0].astype(int))
        cv2.putText(
            image,
            f"{obj_type}, {obj_id}",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    def draw_3d_box(
        self,
        image,
        corners_2d,
        obj_type,
        obj_visibility,
        obj_id="0",
    ):
        """
        Draw 3D bounding box and label on the image.
        Args:
            image (np.ndarray): The image on which to draw.
            corners_2d (np.ndarray): The 2D projected coordinates of the bounding box corners.
            obj_type (str): The object type (e.g., "People", "Vehicle").
        """
        assert corners_2d.shape == (8, 3)

        # Define color and thickness
        color = self.colors.get(obj_type.lower(), self.colors['default'])
        thickness = 2

        # Define the edges of the box (connecting the corners)
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Top four edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Bottom four edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical edges connecting top and bottom
        ]

        # Draw the edges
        for edge in edges:
            pt1 = tuple(corners_2d[edge[0], :2].astype(int))
            pt2 = tuple(corners_2d[edge[1], :2].astype(int))
            cv2.line(image, pt1, pt2, color, thickness)

        # Draw the center of the front face of the box
        center = tuple(((corners_2d[0, :2] + corners_2d[5, :2]) / 2).astype(int))
        cv2.circle(image, center, thickness * 4, color, -1)

        # Add label text (object type) at the top left corner of the box
        label_pos = tuple(corners_2d[0, :2].astype(int))
        cv2.putText(
            image,
            f"{obj_type}, {obj_id}, {100*obj_visibility:.1f}%",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    def process_frame(self, frame):
        obj_list = load_label(self.data_base, frame)
        pose_data = load_pose(self.data_base, frame)

        ##### Sensor #####
        for sensor in self.sensors:
            # Load the label and image data
            img = load_img(self.data_base, sensor, frame)

            # Process each object in the frame
            for obj in obj_list:
                # Get the 3D bounding box corners
                corners_3d_ego = self.get_3d_box_corners(obj)

                # Project the 3D corners to the image plane
                corners_3d_cam = proj_ego_to_sensor(
                    corners_3d_ego, self.sensor_calib[sensor]
                )
                corners_2d, _ = proj_cam_to_img(
                    corners_3d_cam, self.sensor_calib[sensor]
                )

                if len(corners_2d) == 8:
                    # print(f"Object Type: {obj.type}")
                    # print(f"3D Corners in ego: {corners_3d_ego}")
                    # print(f"3D Corners in camera: {corners_3d_cam}")
                    # print(f"2D Corners: {corners_2d}")

                    # Draw the 3D bounding box and label on the image
                    self.draw_3d_box(img, corners_2d, obj.type, obj.visibility, obj.id)

            # Save the modified image
            output_file = os.path.join(self.output_path, sensor, f'{frame}.png')
            # print(f"Saving image to {output_file}")
            cv2.imwrite(output_file, img)

        # ##### UTM BEV #####
        utm_img_size = max(img.shape[0], img.shape[1])
        # utm_img = 255 * np.ones((utm_img_size, utm_img_size, 3), dtype=np.uint8)
        # for obj in obj_list:
        #     corners_3d_ego = self.get_3d_box_corners(obj)
        #     corners_3d_utm = proj_ego_to_ENU(corners_3d_ego, pose_data)
        #     corners_2d_utm = corners_3d_utm[4:, :2]
        #     self.draw_utm_box(utm_img, corners_2d_utm, obj.type, obj.id)

        # # Use pose data x,y to create corners for the utm image
        # ego_corners = np.array(
        #     [
        #         [pose_data.x - 0.3, pose_data.y - 0.3],
        #         [pose_data.x - 0.3, pose_data.y + 0.3],
        #         [pose_data.x + 0.3, pose_data.y + 0.3],
        #         [pose_data.x + 0.3, pose_data.y - 0.3],
        #     ]
        # )
        # self.draw_utm_box(utm_img, ego_corners, "EGO")

        # output_file = os.path.join(self.output_path, "utm", f'{frame}.png')
        # cv2.imwrite(output_file, utm_img)

        ##### EGO BEV #####
        ego_img = 255 * np.ones((utm_img_size, utm_img_size, 3), dtype=np.uint8)
        for obj in obj_list:
            corners_3d_ego = self.get_3d_box_corners(obj)
            corners_2d_ego = corners_3d_ego[4:, :2]
            self.draw_utm_box(ego_img, corners_2d_ego, obj.type, obj.id, utm_range=200)
        ego_corners = np.array(
            [
                [-0.3, -0.3],
                [-0.3, 0.3],
                [0.3, 0.3],
                [0.3, -0.3],
            ]
        )
        self.draw_utm_box(ego_img, ego_corners, "EGO", utm_range=200)

        output_file = os.path.join(self.output_path, "ego", f'{frame}.png')
        cv2.imwrite(output_file, ego_img)

    def run(self):
        # for frame in tqdm(self.frame_list):
        #     # print(f"Processing frame {frame}")
        #     self.process_frame(frame)
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = [
                executor.submit(self.process_frame, frame) for frame in self.frame_list
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()


def main():
    parser = argparse.ArgumentParser(description="Generate Label Visualization")
    parser.add_argument("data_base", type=str, help="Directory of the dataset files")
    parser.add_argument("--frame", type=str, help="Frame to visualize")
    args = parser.parse_args()

    generator = LabelVisualizer(args.data_base)

    if args.frame:
        generator.process_frame(args.frame)
    else:
        generator.run()


if __name__ == "__main__":
    main()
