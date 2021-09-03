# -*- coding:utf-8 -*-

import numpy as np
import cv2

try:
    import config
except Exception:
    from train_code.train_ctpn import config


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    This function resizes the image to the specified width and height.
    It also provides the interpolation method.

    The interpolation method is a way of generating additional pixels in between two pixels of known values.
    The default interpolation method is cv2.INTER_AREA which is used for shrinking an image.
    The interpolation methods are:
        cv2.INTER_NEAREST - a nearest-neighbor interpolation
        cv2.INTER_LINEAR - a bilinear interpolation (used by default)
        cv2.INTER_AREA - resampling using pixel area relation. It may be a preferred method
                         for image decimation, as it gives moire’-free results. But when the
                         image is zoomed, it is similar to the INTER_NEAREST method.
        cv2.INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        cv2.INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

    The function returns the resized image.
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    return image


def gen_anchor(featuresize, scale):
    """
    Generate 9 anchor boxes centered on each pixel of the feature map.
    The size of the anchor at each pixel location is the base size times the scale factor.
    The scale factor is a list of 10 values, one per feature map location.

    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.

    reshape  [HXW][9][4] to [HXWX9][4]
    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchor = np.array([0, 0, 15, 15])
    # center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))


def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : numpy.ndarray
        An N x 4 numpy array, each row is [x1, y1, x2, y2]
    box2 : numpy.ndarray
        An N x 4 numpy array, each row is [x1, y1, x2, y2]
    box1_area : float
        The area of 'box1'
    box2_area : float
        The area of 'box2'

    Returns
    -------
    numpy.ndarray
        An N x M numpy array containing the IoU values.
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    Calculate the overlaps between boxes1 and boxes2.

    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box

    Arguments:
        boxes1 (ndarray): Either :math:`(N, 4)` or :math:`(N, 5)` depending on the value of `clip_boxes`.
        boxes2 (ndarray): Either :math:`(M, 4)` or :math:`(M, 5)` depending on the value of `clip_boxes`.

    Returns:
        ndarray: :math:`(N, M)` overlap between boxes1 and boxes2.
    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps


def bbox_transfrom(anchors, gtboxes):
    """
    Args:
        anchors (ndarray): ndarray of shape (N, 4) for anchor boxes with (x1, y1, x2, y2) format.
        gtboxes (ndarray): ndarray of shape (N, 4) for ground truth boxes with (x1, y1, x2, y2) format.

    Returns:
        ndarray: Regression targets of shape (N, 2). Each row is (Vc, Vh).
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()


def bbox_transfor_inv(anchor, regr):
    """
    Args:
        anchor (ndarray): An array of shape (N x 4) representing anchor boxes
            in (x1, y1, x2, y2) format.
        regr (ndarray): An array of shape (N x 2 x 4) representing predicted
            regression terms for anchors.

    Returns:
        ndarray: N predicted bounding boxes in (x1, y1, x2, y2) format.
    """

    Cya = (anchor[:, 1] + anchor[:, 3]) * 0.5
    ha = anchor[:, 3] - anchor[:, 1] + 1

    Vcx = regr[0, :, 0]
    Vhx = regr[0, :, 1]

    Cyx = Vcx * ha + Cya
    hx = np.exp(Vhx) * ha
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


def clip_box(bbox, im_shape):
    """
    Clip boxes to image boundaries.
    :param bbox: numpy array of shape (N, 4)
    :param im_shape: numpy array of shape (2, )
    :return: numpy array of shape (N, 4)
    """
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox, minsize):
    """
    Filters out bounding boxes that are too small.
    Returns the indices of the bounding boxes that pass the filtering.
    """
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    """
    Args:
        imgsize: [h, w]
        featuresize: the size of each output feature map, e.g. [19, 19]
        scale: the scale factor of the base anchor to the feature map, e.g. [32, 32]
        gtboxes: ground truth boxes in the image, shape of [N, 4].
        stride: the stride of the output feature map.

    Returns:
        labels: label for each anchor, shape of [N, ], -1 for ignore, 0 for background, 1 for object
        bbox_targets: bbox regrssion target for each anchor, shape of [N, 4]
    """
    imgh, imgw = imgsize

    # gen base anchor
    base_anchor = gen_anchor(featuresize, scale)

    # calculate iou
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    # for each GT box corresponds to an anchor which has highest IOU
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IOU > IOU_POSITIVE
    labels[anchor_max_overlaps > config.IOU_POSITIVE] = 1
    # IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps < config.IOU_NEGATIVE] = 0
    # ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0)
        | (base_anchor[:, 1] < 0)
        | (base_anchor[:, 2] >= imgw)
        | (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    # print(len(fg_index))
    if len(fg_index) > config.RPN_POSITIVE_NUM:
        labels[
            np.random.choice(
                fg_index, len(fg_index) - config.RPN_POSITIVE_NUM, replace=False
            )
        ] = -1

    # subsample negative labels
    if not config.OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = config.RPN_TOTAL_NUM - np.sum(labels == 1)
        if len(bg_index) > num_bg:
            # print('bgindex:',len(bg_index),'num_bg',num_bg)
            labels[
                np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)
            ] = -1

    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets], base_anchor


def nms(dets, thresh):
    """
    Parameters
    ----------
    dets : ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    thresh : float
        IOU threshold
    ----------
    Returns
    ----------
    keep : ndarray
        indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# for predict
class Graph:
    """
    Parameters:
        graph (numpy array): A numpy array containing
            the adjacency matrix of the graph

    Returns:
        sub_graphs (list): A list of lists, where each
            sub-list contains nodes that are connected
            to each other.

    Raises:
        TypeError: If graph is not of type numpy.ndarray.

    Examples:
        >>> g = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> print(sub_graphs_connected(g))
        [[0, 1], [2]]

        >>> g = np.array([[0, 1], [1, 0]])
        >>> print(sub_graphs_connected(g))
        [[0], [1]]

        >>> g = np.array([[1, 0], [1, 1]])
        >>> print(sub_graphs_connected(g))
        [[0], [1]]

        >>> g = np.array([[1, 1], [1, 1]])
        >>> print(sub_graphs_connected(g))
        [[0, 1]]

        >>> g = np.array([[0, 0], [0, 0]])
        >>> print(sub_graphs_connected(g))
        []

        >>> g = np.array([[1, 0], [0, 1]])
        >>> print(sub_graphs_connected(g))
        [[0], [1]]

        >>> g = np.array([[1, 1], [0, 0]])
        >>> print(sub_graphs_connected(g))
        [[0], [1]]

        >>> g = np.array([[1, 1], [1, 0]])
        >>> print(sub_graphs_connected(g))
        [[0], [1]]
    """

    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9
    MAX_HORIZONTAL_GAP = 60
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6


class TextProposalGraphBuilder:
    """
    text_proposals:
    Nx4 numpy array
    N is the number of boxes
    Each row is [x1, y1, x2, y2]

    scores:
        N numpy array
        N is the number of boxes
        Each element is a score for each box

    im_size:
        2-tuple, (width, height)
        width and height are integers indicating the size of the image.

    boxes_table:
        A list containing N lists.
        Each list is a list of indices of text proposals that belong to the same horizontal line.
        The first element in each list is the index of the horizontal line.
        The rest of the elements are the indices of the text proposals that belong to this line.

    heights:
        N numpy array.
        Each element indicates the height of each box.
    """

    def get_successions(self, index):
        """
        1. Get the bounding box of the text line from the text_proposals.
        2. Get all the bounding boxes in the same row as the bounding box of the text line.
        3. For each bounding box in the same row, check if it's a successor or not.
        4. If it is a successor, add it to the list of successors.
        5. Return a list of successors.
        """
        box = self.text_proposals[index]
        results = []
        for left in range(
            int(box[0]) + 1,
            min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1]),
        ):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        """
        Inputs:
        index: The box to look for a precursor of.

        Returns:
        A list of indices corresponding to the previous boxes which are connected
        to the current box. In other words, these previous boxes refer to lines that
        overlap with the current line.
        """
        box = self.text_proposals[index]
        results = []
        for left in range(
            int(box[0]) - 1,
            max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1,
            -1,
        ):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        """
        Returns whether one node is the successor of the other

        Parameters:
            index: int, the index of the node
           succession_index: int, the index of the possible successor node

        Returns:
            is_succession_node: bool, whether one node is the successor of the other
        """
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        """
        If both bounding boxes are present in the ground truth text line bounding box list
        and the overlap is larger than a minimum threshold AND
        their size similarity is smaller than a given value, return True.
        """

        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return (
            overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS
            and size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM
        )

    def build_graph(self, text_proposals, scores, im_size):
        """
        # Inputs:
        - `text_proposals`: a list of text proposals.
        - `scores`: a list of text proposals' scores.
        - `im_size`: a tuple of (image_height, image_width).

        # Outputs:
        - `graph`: a `Graph` instance, holding a graph representation of text proposals.

        # Notes:
        - This function builds a graph from the set of given text proposals.
        - Each node in the graph corresponds to a text proposal.
        - Each text proposal has a set of successors.
        - Each node is a `Succession` instance, which holds the following data:
            - `index`: the index of the text proposal.
            - `score`: the score of the text proposal.
            - `successions`: a list of indexes of successor text proposals.
            - `precursors`: a list of indexes of precursor text proposals.
            - `mask_index`: the index of the mask that the text proposal belongs to.
        """
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
        return Graph(graph)


class TextProposalConnectorOriented:
    """
    Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        # len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        1. First, it finds the text proposals that are connected into text lines.
        2. Then, it finds the average value of the center of the text line and the height of the text line.
        3. Finally, it fits a straight line according to the center and height of the text line,
           and then calculates the b value of the straight line.
        """
        # tp=text proposal
        # First of all, it is to build a picture, and get the small boxes constituted by the text line
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[
                list(tp_indices)
            ]  # All small boxes for each text line
            X = (
                text_line_boxes[:, 0] + text_line_boxes[:, 2]
            ) / 2  # Find the x and y coordinates of the center of each small box
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            # Polynomial fitting, fitting a straight line according to the center store found before (least squares)
            z1 = np.polyfit(X, Y, 1)

            x0 = np.min(
                text_line_boxes[:, 0]
            )  # The minimum x coordinate of the text line
            x1 = np.max(text_line_boxes[:, 2])  # Maximum x coordinate of text line

            offset = (
                text_line_boxes[0, 2] - text_line_boxes[0, 0]
            ) * 0.5  # Half the width of the small box

            # Fit a straight line with the point on the upper left corner of all the small boxes,
            # and then calculate the y coordinate corresponding to the extreme left and right
            # of the x coordinate of the text line
            lt_y, rt_y = self.fit_y(
                text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset
            )
            # Fit a straight line with the point at the lower left corner of all the small boxes,
            # and then calculate the y coordinate corresponding to the extreme left and right
            # of the x coordinate of the text line
            lb_y, rb_y = self.fit_y(
                text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset
            )

            # Find the average value of all the small box scores as the average value of the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(
                lt_y, rt_y
            )  # The small value of the y coordinate of the line segment at the top of the text line
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(
                lb_y, rb_y
            )  # The maximum value of the y coordinate of the line segment at the bottom of the text line
            text_lines[index, 4] = score  # Text line score
            text_lines[index, 5] = z1[
                0
            ]  # K, b of a straight line fitted according to the center point
            text_lines[index, 6] = z1[1]
            height = np.mean(
                (text_line_boxes[:, 3] - text_line_boxes[:, 1])
            )  # Average height of small box
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            # According to the height and the center line of the text line,
            # find the b value of the upper and lower lines of the text line
            b1 = line[6] - line[7] / 2
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # upper left
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # upper right
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # lower left
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # lower right
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # Text line width

            fTmp0 = y3 - y1  # Text line height
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # Make compensation
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
