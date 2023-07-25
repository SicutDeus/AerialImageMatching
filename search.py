import cv2
import numpy as np
import time
import config


class LocationSearch:
    def __init__(self):
        self.search_method = cv2.AKAZE.create()
        self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMINGLUT)
        self.last_points = []
        self.is_focus = False

    def crop_target(self, target_img):
        h, w = target_img.shape[:2]
        xmin = w
        xmax = 0
        ymin = h
        ymax = 0
        for point in self.last_points:
            xmax = max(point[0][0], xmax)
            xmin = min(point[0][0], xmin)
            ymax = max(point[0][1], ymax)
            ymin = min(point[0][1], ymin)

        dist = 0.5 * max(xmax - xmin, ymax - ymin)
        x1 = round(max(xmin - dist, 0))
        x2 = round(min(xmax + dist, w))
        y1 = round(max(ymin - dist, 0))
        y2 = round(min(ymax + dist, h))

        # print(x1,x2,y1,y2)
        crop_img = target_img[y1:y2, x1:x2]
        return crop_img, x1, y1

    def get_keypoints_and_descriptors(self, source_img, target_img, offset_x, offset_y):
        #t = time.time()
        keypoints_source, descriptors_source = self.search_method.detectAndCompute(source_img, None)
        keypoints_target, descriptors_target = self.search_method.detectAndCompute(target_img, None)
        if offset_x is not None:
            for keypoint in keypoints_target:
                keypoint.pt = (keypoint.pt[0] + offset_x, keypoint.pt[1] + offset_y)
        #print(time.time() - t)
        return keypoints_source, descriptors_source, keypoints_target, descriptors_target

    def find_matches(self, descriptors_source, descriptors_target, k):
        matches = self.matcher.knnMatch(descriptors_source, descriptors_target, k=k)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        return good_matches

    def search(self, source_img, target_img):
        if self.is_focus:
            crop_target, offset_x, offset_y = self.crop_target(target_img)
            kp_s, des_s, kp_t, des_t = self.get_keypoints_and_descriptors(source_img, crop_target, offset_x, offset_y)
        else:
            kp_s, des_s, kp_t, des_t = self.get_keypoints_and_descriptors(source_img, target_img, None, None)
        good_matches = self.find_matches(des_s, des_t, 2)

        if len(good_matches) >= config.MIN_MATCH_COUNT:
            src_pts = np.float32([kp_s[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = source_img.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            self.last_points = np.int32(dst)

            result_img = cv2.polylines(target_img, [np.int32(dst)], True, 255, 5, cv2.LINE_AA)

            M = cv2.moments(self.last_points)

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            result_img = cv2.circle(result_img, (cx, cy), 4, (0, 255, 255), 10)
            self.is_focus = True
        else:
            print("Not enough matches are found - %d/%d" % (len(good_matches), self.MIN_MATCH_COUNT))
            matches_mask = None
            result_img = target_img
            self.is_focus = False

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)

        output_image = cv2.drawMatches(source_img, kp_s, result_img[:, :, :3], kp_t, good_matches, None, **draw_params)
        return output_image
