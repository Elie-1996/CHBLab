import math
import cv2
import numpy as np
import pandas as pd


info = 'File Generates Scanpath. All you need to do is run the file and it should work. ' \
    'All Parameters are specified in Utils and some additional ones below' \
    'In addition, this file calculates the entropy in each bin, although it may be incomplete after second-checking' \
    'Note: Entropy was eventually not used in the study. I\'ll keep the function available, ' \
    'if needed you can build on and fix it then test.'


# TODO: Suggestion - turn all parameters into a class with ENUMS to avoid confusion
HORIZONTAL_BINS, VERTICAL_BINS = 9, 9  # Placing -1 on either HORIZONTAL or VERTICAL bins will give you exact
DRAW_CENTER_CONNECTIONS = True
CONNECT_CENTERS_WITH_ARROWS_OR_LINES = False  # True = Arrows, False = Lines.

MILLISECONDS_WAIT_TIME_BEFORE_CLOSING_PLOT = 1000*20
MILLISECONDS_BETWEEN_FRAMES = 2


SHOULD_SAVE_GIF, NUMBER_OF_FRAMES_TO_SAVE = True, 200
ENTROPY_SHOULD_COMBINE_BINS = False  # predefined bins are binned together when calculating entropy=> THIS DOES NOT WORK


def preprocess_combinations(amounts_passage, xSize, ySize):
    if not ENTROPY_SHOULD_COMBINE_BINS:
        return amounts_passage  # perform regular binning
    else:  # combine bins together according to combinations
        combinations = [
            [(i, i) for i in range(9)],
            [(1, 1), (1, 2)],
            [(2, 2), (3, 3)]
        ]
        # transform to correct key:
        comb_size = combinations.__len__()
        for i in range(comb_size):
            group = combinations[i]
            size = group.__len__()
            for j in range(size):
                newX = int((2 * group[j][0] + 1) * (xSize / HORIZONTAL_BINS) / 2)
                newY = int((group[j][1] + 1) * (ySize / VERTICAL_BINS) / 2)
                key = (newX, newY)
                print(key)
                combinations[i][j] = key

        print("------")
        for a in amounts_passage.keys():
            print(a)
        print("------------")

        for group in combinations:
            group_into_bin = group[0]
            size = group.__len__()
            for i in range(1, size):
                current_bin = group[i]
                amounts_passage[group_into_bin] += amounts_passage[current_bin]
                amounts_passage[current_bin] = 0
        return amounts_passage


def calculate_distribution(amounts_passage, sum_passage, xSize, ySize):
    distributionX = []
    amounts_passage = preprocess_combinations(amounts_passage, xSize, ySize)
    for value in amounts_passage.values():
        distributionX.append(value / sum_passage)
    return distributionX


def calculate_entropy(distributionX):
    entropyOfX = 0
    for percentage in distributionX:
        if percentage == 0:
            continue
        entropyOfX += percentage * math.log2(percentage)
    entropyOfX = -entropyOfX
    return entropyOfX


def preprocess_data_frame(data_frame):
    # make all time start from 0
    normalize_time = data_frame['start_timestamp'].iloc[0]
    data_frame['start_timestamp'] -= normalize_time

    # discard all points that fall out of surface
    data_frame['norm_pos_x'] = data_frame[data_frame['norm_pos_x'] >= 0]['norm_pos_x']
    data_frame['norm_pos_x'] = data_frame[data_frame['norm_pos_x'] <= 1]['norm_pos_x']
    data_frame['norm_pos_y'] = data_frame[data_frame['norm_pos_y'] >= 0]['norm_pos_y']
    data_frame['norm_pos_y'] = data_frame[data_frame['norm_pos_y'] <= 1]['norm_pos_y']

    # divide dataframe according to number of bins
    data_frame['norm_pos_x'] = pd.cut(data_frame['norm_pos_x'], HORIZONTAL_BINS)
    data_frame['norm_pos_y'] = pd.cut(data_frame['norm_pos_y'], VERTICAL_BINS)

    return data_frame


def create_scanpath_images(data_frame):
    from Utils import height, width, subject_times, OFFSET_X, OFFSET_Y
    num_rows = len(data_frame)
    scanpath = []  # This will contain all the timeframes for relevant sampled points
                # and mid-point locations in the bins. see comment below for explanation.

    idx = 0
    while True:
        if idx >= num_rows or data_frame['start_timestamp'].iloc[idx] > subject_times[1]:
            break
        if data_frame['on_surf'].iloc[idx] and subject_times[0] <= data_frame['start_timestamp'].iloc[idx]:
            x, y = data_frame['norm_pos_x'].iloc[idx], data_frame['norm_pos_y'].iloc[idx]
            x, y = x.mid, y.mid
            finalx, finaly = x * width + OFFSET_X, (1 - y) * height + OFFSET_Y
            if (0 <= finalx <= width) and (0 <= finaly <= height):
                scanpath.append([x * width + OFFSET_X,
                                 (1 - y) * height + OFFSET_Y,
                                 data_frame['start_timestamp'].iloc[idx],
                                 data_frame['start_timestamp'].iloc[idx + 1] - data_frame['start_timestamp'].iloc[idx]])
                # As you can see, we append 4 values: (point_x, point_y, start_time, end_time). where
                # (point_x, point_y) refer to the middle point of the circle formed in the end-result.
                # and (start_time, end_time) refer to the times of these dots.
        idx += 1

    scanpath = np.asarray(scanpath)
    return scanpath


# Function helps calculate the entropy
def markAdditionalPassOnBin(amounts_passage, key, sum_passage):
    if amounts_passage.get(key) is None:
        amounts_passage[key] = 0
    amounts_passage[key] += 1
    return amounts_passage[key], sum_passage + 1


# draws the lines or arrows between two centers of different bins.
def connectCenters(scanpath, i, fixation, frame):
    left_ind, right_ind = 0, 1
    left_color,  right_color = [0, 0, 0], [0, 0, 0]
    prev_fixation = scanpath[i - 1].astype(int)
    new_color = (i * 4) % 256

    if prev_fixation[0] > fixation[0] or prev_fixation[1] > fixation[1]:
        left_color[left_ind] = new_color
        left_color[right_ind] = 0
        if CONNECT_CENTERS_WITH_ARROWS_OR_LINES:
            # For arrow ->
            cv2.arrowedLine(frame, (prev_fixation[0], prev_fixation[1]), (fixation[0], fixation[1]),
                            left_color, thickness=3, shift=0)
        else:
            cv2.line(frame, (prev_fixation[0], prev_fixation[1]), (fixation[0], fixation[1]),
                 left_color, thickness=3, shift=0)
    else:
        right_color[right_ind] = new_color
        right_color[left_ind] = 0
        if CONNECT_CENTERS_WITH_ARROWS_OR_LINES:
            # For arrow <-
            cv2.arrowedLine(frame, (prev_fixation[0], prev_fixation[1]), (fixation[0], fixation[1]),
                            right_color, thickness=3, shift=0)
        else:
            cv2.line(frame, (prev_fixation[0], prev_fixation[1]), (fixation[0], fixation[1]),
                 right_color, thickness=3, shift=0)


# without this function, the plot will be too large for the screens.
def normalizeFrameSizesInPlot(toPlot, plotMaxDim):
    for i in range(len(toPlot)):
        h, w, _ = np.shape(toPlot[i])
        h, w = float(h), float(w)
        if h > w:
            w = (plotMaxDim / h) * w
            h = plotMaxDim
        else:
            h = (plotMaxDim / w) * h
            w = plotMaxDim
        h, w = int(h), int(w)
        toPlot[i] = cv2.resize(toPlot[i], (w, h), interpolation=cv2.INTER_CUBIC)
    return toPlot


# draw the rectangles of the bins.
def drawBins(toPlot, frame):
    from Utils import height, width
    overlay = frame.copy()
    for current_bin in range(HORIZONTAL_BINS):
        overlay = overlay.copy()
        fixed_y = int(current_bin * height / HORIZONTAL_BINS)
        xy1 = (0, fixed_y)
        xy2 = (width - 1, fixed_y)
        black = [0, 0, 0]
        cv2.line(overlay, xy1, xy2, black, 3)
    toPlot.append(overlay)

    for current_bin in range(VERTICAL_BINS):
        overlay = overlay.copy()
        fixed_x = int(current_bin * width / VERTICAL_BINS)
        xy1 = (fixed_x, 0)
        xy2 = (fixed_x, height - 1)
        black = [0, 0, 0]
        cv2.line(overlay, xy1, xy2, black, 3)
    toPlot.append(overlay)
    return toPlot


def generate_scanpath(drawNumberFixations=False, plotMaxDim=1024):
    ''' This functions uses cv2 standard library to visualize the scanpath
        of a specified stimulus.
        Depending on the monitor or the image dimensions, it could be convenient to
        resize the images before to plot them. In such a case, user could indicate in
        the additional argument plotMaxDim=500 to set, for example, the maximum
        dimension to 500. By default, images are not resized.'''
    # Loading Data
    from Utils import QUESTION_IDX, img_path, SUBJECT_KEY, subject_path, height, width
    img = cv2.imread(img_path)
    df = pd.read_csv(subject_path)

    # Preprocessing
    assert (HORIZONTAL_BINS >= 1 and VERTICAL_BINS >= 1)
    df = preprocess_data_frame(df)
    scanpath = create_scanpath_images(df)

    print("Preprocess data")

    toPlot = [img, ]  # List of images to plot. Need a non-empty list because later we assume to always build on the
                      # previous image.

    amounts_passage = {}
    sum_passage = 0

    # Creating scanpath
    for i in range(np.shape(scanpath)[0]):

        fixation = scanpath[i].astype(int)
        key = (fixation[0], fixation[1])
        amounts_passage[key], sum_passage = markAdditionalPassOnBin(amounts_passage, key, sum_passage)

        frame = np.copy(toPlot[-1]).astype(np.uint8)
        overlay = frame.copy()
        cv2.circle(frame, (fixation[0], fixation[1]), 10, (0, 204, 0), -1)
        alpha = 0  # when set to 0 transparency of circles is 100%
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        if drawNumberFixations:
            cv2.putText(frame, str(i + 1), (fixation[0], fixation[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                        thickness=2)
        if DRAW_CENTER_CONNECTIONS and i > 0:
            connectCenters(scanpath, i, fixation, frame)
        # if animation is required, frames are attached in a sequence
        # if not animation is required, older frames are removed
        toPlot.append(frame)

    for current_key in amounts_passage:
        overlay = frame.copy()
        cv2.circle(overlay, current_key, 25 + amounts_passage.get(current_key), (0, 204, 0), -1)
        alpha = 0.4  # when set to 0 transparency of circles is 100%
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    toPlot.append(frame)

    # draw bins
    toPlot = drawBins(toPlot, frame)
    toPlot = normalizeFrameSizesInPlot(toPlot, plotMaxDim)

    print("Now its imshow")
    import imageio
    if SHOULD_SAVE_GIF:
        # imageio.mimsave('./scanpath.gif', toPlot[:NUMBER_OF_FRAMES_TO_SAVE])
        for repeat in range(150):
            toPlot.append(toPlot[-1])
        imageio.mimsave(
            './' + SUBJECT_KEY + '_question' + str(QUESTION_IDX + 1) + '_scanpath.gif',
            toPlot[:])
    print("saving complete")
    # calculate the entropy
    distributionX = calculate_distribution(amounts_passage, sum_passage, width, height)
    entropyOfX = calculate_entropy(distributionX)
    print("Entropy of data: " + str(entropyOfX))

    for i in range(len(toPlot)):
        cv2.imshow('Scanpath of Subject ' + SUBJECT_KEY.split('_')[0] + ' solving question ' + str(QUESTION_IDX + 1),
                   toPlot[i])
        cv2.waitKey(MILLISECONDS_BETWEEN_FRAMES)
    print("Wait Key for " + str(MILLISECONDS_WAIT_TIME_BEFORE_CLOSING_PLOT*0.001) + " seconds")
    cv2.waitKey(MILLISECONDS_WAIT_TIME_BEFORE_CLOSING_PLOT)
    cv2.destroyAllWindows()
    print("Finish video")


generate_scanpath()
