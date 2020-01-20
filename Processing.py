# -*- coding: utf-8 -*-
from skimage import exposure
import numpy as np
import cv2


def max_lines(lines1, lines2, flag):
    x_list = [lines1[0], lines1[2], lines2[0], lines2[2]]
    y_list = [lines1[1], lines1[3], lines2[1], lines2[3]]
    if flag == 1:
        max_x = max(x_list)
        min_x = min(x_list)
        max_flag = x_list.index(max_x)
        min_flag = x_list.index(min_x)
        result = [x_list[min_flag], y_list[min_flag], x_list[max_flag], y_list[max_flag]]
        return result
    if flag == 0:
        max_y = max(y_list)
        min_y = min(y_list)
        max_flag = y_list.index(max_y)
        min_flag = y_list.index(min_y)
        result = [x_list[min_flag], y_list[min_flag], x_list[max_flag], y_list[max_flag]]
        return result


def mesh_lines(lines, point_x, point_y, flag, threshold=0.05):
    final_lines, temp_list, len_list = [], [], []
    k, j = 0, 0
    for i in range(len(lines)):
        array_longi = np.array([lines[i][0][2] - lines[i][0][0], lines[i][0][3] - lines[i][0][1]])
        array_trans = np.array([lines[i][0][2] - point_x, lines[i][0][3] - point_y])
        array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))
        array_temp = array_longi.dot(array_temp)
        distance = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))
        len_list.append(distance)
    while len(temp_list) != len(lines):
        if k not in temp_list:
            final_lines.append(lines[k])
            temp_list.append(k)
            for i in range(len(lines)):
                if i not in temp_list and np.abs((len_list[k] - len_list[i]) / len_list[k]) < threshold:
                    final_lines[j][0] = max_lines(lines[i][0], final_lines[j][0], flag)
                    temp_list.append(i)
            j = j + 1
        k = k + 1
    return final_lines


def choose_lines(lines, point_x, point_y, threshold=3):
    if len(lines) > 1:
        final_lines, len_list = [], []
        len_lines1 = pow(lines[0][0][3] - lines[0][0][1], 2) + pow(lines[0][0][2] - lines[0][0][0], 2)
        len_lines2 = pow(lines[1][0][3] - lines[1][0][1], 2) + pow(lines[1][0][2] - lines[1][0][0], 2)
        for i in range(len(lines)):
            array_longi = np.array([lines[i][0][2] - lines[i][0][0], lines[i][0][3] - lines[i][0][1]])
            array_trans = np.array([lines[i][0][2] - point_x, lines[i][0][3] - point_y])
            array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))
            array_temp = array_longi.dot(array_temp)
            distance = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))
            len_list.append(distance)
        if (len_list[0] > len_list[1] and len_lines1 / len_lines2 > threshold) or (len_list[0] < len_list[1] and len_lines2 / len_lines1 > threshold):
            final_lines.append(lines[len_list.index(max(len_list))])
        else:
            final_lines.append(lines[len_list.index(min(len_list))])
        return final_lines
    else:
        return lines


def get_point(line1, line2, line3, line4):
    xx, yy = [], []
    lines = line1 + line2 + line3 + line4
    for line in lines:
        for x1, y1, x2, y2 in line:
            xx.append((x1 + x2) / 2)
            yy.append((y1 + y2) / 2)
    return [int(np.sum(xx) / len(lines)), int(np.sum(yy) / len(lines))]


def delete_lines(lines, point_x, point_y, threshold=2):
    if len(lines) > threshold:
        len_list, final_lines = [], []
        test_lines = lines
        for i in range(len(lines)):
            array_longi = np.array([lines[i][0][2] - lines[i][0][0], lines[i][0][3] - lines[i][0][1]])
            array_trans = np.array([lines[i][0][2] - point_x, lines[i][0][3] - point_y])
            array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))
            array_temp = array_longi.dot(array_temp)
            distance = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))
            len_list.append(distance)
        for i in range(threshold):
            final_lines.append(test_lines[len_list.index(min(len_list))])
            del test_lines[len_list.index(min(len_list))]
            len_list.remove(min(len_list))
        return final_lines
    else:
        return lines


def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]
    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 is None:
        x = x3
        y = k1 * x3 + b1
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def drawing_pic(drawing, lines, line_color):
    line_thick = 3
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(drawing, (x1, y1), (x2, y2), line_color, line_thick)
    return drawing


def getcut(img, name):
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_edge = 50
    max_edge = 100
    min_slope = 0.2
    max_slope = 5.5
    min_line_len = 150
    max_line_gap = 20
    x, y = [], []
    left_lines, right_lines = [], []
    up_lines, down_lines = [], []
    img_pre = cv2.medianBlur(img, 5)
    img_gray = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    img_bin, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    dilation = cv2.dilate(img_bin, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    img_bin = cv2.dilate(erosion, element2, iterations=3)
    img_edges = cv2.Canny(img_bin, min_edge, max_edge)
    cv2.imencode('.jpg', img_pre)[1].tofile(name + '_mid_result.jpg')
    cv2.imencode('.jpg', img_gray)[1].tofile(name + '_gray_result.jpg')
    cv2.imencode('.jpg', img_bin)[1].tofile(name + '_bin_result.jpg')
    cv2.imencode('.jpg', img_edges)[1].tofile(name + '_edges_result.jpg')
    lines = cv2.HoughLinesP(img_edges, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    drawing = np.zeros((img_edges.shape[0], img_edges.shape[1], 3), dtype=np.uint8)
    lines_all = drawing_pic(drawing, lines, [0, 0, 255])
    cv2.imencode('.jpg', lines_all)[1].tofile(name + '_lines_all_result.jpg')
    for line in lines:
        for x1, y1, x2, y2 in line:
            x.append((x1 + x2) / 2)
            y.append((y1 + y2) / 2)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2 or np.abs((y2 - y1) / (x2 - x1)) > max_slope:
                if (x1 + x2) / 2 > np.sum(x) / len(lines):
                    right_lines.append(line)
                if (x1 + x2) / 2 < np.sum(x) / len(lines):
                    left_lines.append(line)
            if x1 != x2 and np.abs((y2 - y1) / (x2 - x1)) < min_slope:
                if (y1 + y2) / 2 > np.sum(y) / len(lines):
                    up_lines.append(line)
                if (y1 + y2) / 2 < np.sum(y) / len(lines):
                    down_lines.append(line)
    test_line = [lines[0]]
    if len(left_lines) <= 0:
        left_lines = test_line
        left_lines[0][0] = [img_edges.shape[0], 0, 0, 0]
    if len(right_lines) <= 0:
        right_lines = test_line
        right_lines[0][0] = [img_edges.shape[0], img_edges.shape[1], img_edges.shape[0], 0]
    if len(up_lines) <= 0:
        up_lines = test_line
        up_lines[0][0] = [img_edges.shape[0], 0, img_edges.shape[0], img_edges.shape[1]]
    if len(down_lines) <= 0:
        down_lines = test_line
        down_lines[0][0] = [0, 0, 0, img_edges.shape[1]]
    left_lines = mesh_lines(left_lines, np.sum(x) / len(lines), np.sum(y) / len(lines), 0)
    right_lines = mesh_lines(right_lines, np.sum(x) / len(lines), np.sum(y) / len(lines), 0)
    up_lines = mesh_lines(up_lines, np.sum(x) / len(lines), np.sum(y) / len(lines), 1)
    down_lines = mesh_lines(down_lines, np.sum(x) / len(lines), np.sum(y) / len(lines), 1)
    lines_mesh = drawing_pic(drawing, left_lines, [0, 255, 0])
    lines_mesh = drawing_pic(drawing, right_lines, [0, 255, 0])
    lines_mesh = drawing_pic(drawing, up_lines, [0, 255, 0])
    lines_mesh = drawing_pic(drawing, down_lines, [0, 255, 0])
    cv2.imencode('.jpg', lines_mesh)[1].tofile(name + '_lines_mesh_result.jpg')
    [xx, yy] = get_point(left_lines, right_lines, up_lines, down_lines)
    left_lines = delete_lines(left_lines, xx, yy)
    right_lines = delete_lines(right_lines, xx, yy)
    up_lines = delete_lines(up_lines, xx, yy)
    down_lines = delete_lines(down_lines, xx, yy)
    lines_pre = drawing_pic(drawing, left_lines, [255, 255, 0])
    lines_pre = drawing_pic(drawing, right_lines, [255, 255, 0])
    lines_pre = drawing_pic(drawing, up_lines, [255, 255, 0])
    lines_pre = drawing_pic(drawing, down_lines, [255, 255, 0])
    cv2.imencode('.jpg', lines_pre)[1].tofile(name + '_lines_pre_result.jpg')
    left_line_pre = choose_lines(left_lines, xx, yy)
    right_line_pre = choose_lines(right_lines, xx, yy)
    up_line_pre = choose_lines(up_lines, xx, yy)
    down_line_pre = choose_lines(down_lines, xx, yy)
    [x_up_left, y_up_left] = cross_point(up_line_pre[0][0], left_line_pre[0][0])
    [x_up_right, y_up_right] = cross_point(up_line_pre[0][0], right_line_pre[0][0])
    [x_down_left, y_down_left] = cross_point(down_line_pre[0][0], left_line_pre[0][0])
    [x_down_right, y_down_right] = cross_point(down_line_pre[0][0], right_line_pre[0][0])
    lines_finall = drawing_pic(drawing, left_line_pre, [255, 0, 255])
    lines_finall = drawing_pic(drawing, right_line_pre, [255, 0, 255])
    lines_finall = drawing_pic(drawing, up_line_pre, [255, 0, 255])
    lines_finall = drawing_pic(drawing, down_line_pre, [255, 0, 255])
    cv2.circle(lines_finall, (int(x_up_left), int(y_up_left)), 10, (0, 255, 255), 4)
    cv2.circle(lines_finall, (int(x_up_right), int(y_up_right)), 10, (0, 255, 255), 4)
    cv2.circle(lines_finall, (int(x_down_left), int(y_down_left)), 10, (0, 255, 255), 4)
    cv2.circle(lines_finall, (int(x_down_right), int(y_down_right)), 10, (0, 255, 255), 4)
    cv2.imencode('.jpg', lines_finall)[1].tofile(name + '_lines_result.jpg')
    return [x_up_left, y_up_left], [x_up_right, y_up_right], [x_down_left, y_down_left], [x_down_right, y_down_right]


def getshape(img, cut):
    dis_k = 1.5
    for point in cut:
        point[0] = int(point[0])
        point[1] = int(point[1])
    pts1 = np.float32([cut[2], cut[3], cut[1], cut[0]])
    if (cut[1][0] - cut[0][0]) / (cut[0][1] - cut[2][1]) > dis_k:
        pts2 = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (1920, 1080))
    else:
        pts2 = np.float32([[0, 0], [1440, 0], [1440, 1080], [0, 1080]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (1440, 1080))
    return result


def getcolor(img):
    gamma_img = exposure.adjust_gamma(img, 1.6)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    gamma_img = cv2.filter2D(gamma_img, -1, kernel=kernel)
    result = cv2.fastNlMeansDenoisingColored(gamma_img, None, 10, 10, 7, 21)
    return result


def main_programe(filename, name):
    img_1 = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    img_2 = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    cut = getcut(img_1, name)
    cv2.circle(img_1, (int(cut[0][0]), int(cut[0][1])), 10, (0, 255, 255), 4)
    cv2.circle(img_1, (int(cut[1][0]), int(cut[1][1])), 10, (0, 255, 255), 4)
    cv2.circle(img_1, (int(cut[2][0]), int(cut[2][1])), 10, (0, 255, 255), 4)
    cv2.circle(img_1, (int(cut[3][0]), int(cut[3][1])), 10, (0, 255, 255), 4)
    shape = getshape(img_2, cut)
    color = getcolor(shape)
    return img_1, shape, color


if __name__ == "__main__":
    file_name = 'testphoto0'
    final_img = main_programe(file_name + '.jpg', file_name)
    cv2.imencode('.jpg', final_img[0])[1].tofile(file_name + '_check.jpg')
    cv2.imencode('.jpg', final_img[1])[1].tofile(file_name + '_shape.jpg')
    cv2.imencode('.jpg', final_img[2])[1].tofile(file_name + '_result.jpg')
    print(file_name + "  处理完成")
