import os
import random

import cv2
import numpy as np
import pandas as pd
import math
from Utils import background_images, input_fixations_directory, subjects_dict, strategy, WIDTHS, HEIGHTS


SHOULD_SAVE_GIF, NUMBER_OF_FRAMES_TO_SAVE = True, 200
# extra_text = 'min_'
extra_text = 'whole_'
# extra_text = 'strategy' + str(strategy) + '_'  # don't forget to add strategy index/tag!
# extra_text = 'IGNORE_THIS_RESULT' + str(strategy) + '_'
QUESTION_IDX = 2
SUBJECT_KEY = '001'  # take the key from subjects_dict (imported above :) )
HORIZONTAL_BINS, VERTICAL_BINS = 9, 9  # Placing -1 on either HORIZONTAL or VERTICAL bins will give you exact
                                       # coordinates (no bins)
ENTROPY_SHOULD_COMBINE_BINS = False  # predefined bins are binned together when calculating entropy => THIS DOES NOT WORK
# OFFSET_X, OFFSET_Y = -250, -46
OFFSET_X, OFFSET_Y = 0, 0


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
                newX = int((2*group[j][0] + 1) * (xSize/HORIZONTAL_BINS) / 2)
                newY = int((group[j][1] + 1) * (ySize/VERTICAL_BINS) / 2)
                key = (newX, newY)
                print(key)
                combinations[i][j] = key

        print ("------")
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


def scanpath(animation=True, wait_time=30000, putLines=True, putNumbers=False, plotMaxDim=1024):
    ''' This functions uses cv2 standard library to visualize the scanpath
        of a specified stimulus.
        By default, one random scanpath is chosen between available subjects. For
        a specific subject, it is possible to specify its id on the additional
        argument subject=id.
        It is possible to visualize it as an animation by setting the additional
        argument animation=True.
        Depending on the monitor or the image dimensions, it could be convenient to
        resize the images before to plot them. In such a case, user could indicate in
        the additional argument plotMaxDim=500 to set, for example, the maximum
        dimension to 500. By default, images are not resized.'''
    ## Loading Data
    img_path = os.path.join('..', 'Heatmap', background_images[QUESTION_IDX])
    subject_path = os.path.join('..', input_fixations_directory, SUBJECT_KEY + "_fixations.csv")
    img = cv2.imread(img_path)
    height, width, layers = img.shape
    from Utils import WIDTHS, HEIGHTS
    height = HEIGHTS[QUESTION_IDX]
    width = WIDTHS[QUESTION_IDX]
    size = (width, height)
    df = pd.read_csv(subject_path)

    ## Init vars
    scanpath = []

    ## Preprocessing
    normalize_time = df['start_timestamp'].iloc[0]
    df['start_timestamp'] -= normalize_time

    df['norm_pos_x'] = df[df['norm_pos_x'] >= 0]['norm_pos_x']
    df['norm_pos_x'] = df[df['norm_pos_x'] <= 1]['norm_pos_x']
    df['norm_pos_y'] = df[df['norm_pos_y'] >= 0]['norm_pos_y']
    df['norm_pos_y'] = df[df['norm_pos_y'] <= 1]['norm_pos_y']


    should_draw_bins = HORIZONTAL_BINS >= 1 and VERTICAL_BINS >= 1
    if should_draw_bins:
        df['norm_pos_x'] = pd.cut(df['norm_pos_x'], HORIZONTAL_BINS)
        df['norm_pos_y'] = pd.cut(df['norm_pos_y'], VERTICAL_BINS)

    subject_times = subjects_dict[SUBJECT_KEY][QUESTION_IDX]
    num_rows = len(df)

    print("Preprocess data")
    idx = 0
    while True:
        if idx >= num_rows or df['start_timestamp'].iloc[idx] > subject_times[1]:
            break
        if df['on_surf'].iloc[idx] and subject_times[0] <= df['start_timestamp'].iloc[idx]:
            x, y = df['norm_pos_x'].iloc[idx], df['norm_pos_y'].iloc[idx]
            if should_draw_bins:
                x, y = x.mid, y.mid
            finalx, finaly = x * size[0] + OFFSET_X, (1 - y) * size[1] + OFFSET_Y
            if finalx >= 0 and finalx <= size[0] and finaly >= 0 and finaly <= size[1]:
                scanpath.append([x * size[0] + OFFSET_X, (1 - y) * size[1] + OFFSET_Y,
                         df['start_timestamp'].iloc[idx],
                         df['start_timestamp'].iloc[idx + 1] - df['start_timestamp'].iloc[idx]])
        idx += 1

    toPlot = [img, ]  # look, it is a list!
    scanpath = np.asarray(scanpath)
    left_color = [0, 0, 0]
    right_color = [0, 0, 0]

    ## Creating scanpath
    print("Fixations are ready, start making the scanpath")
    left_ind = 0
    right_ind = 1

    amounts_passage = {}
    sum_passage = 0
    for i in range(np.shape(scanpath)[0]):

        fixation = scanpath[i].astype(int)

        frame = np.copy(toPlot[-1]).astype(np.uint8)

        key = (fixation[0], fixation[1])
        if amounts_passage.get(key) is None:
            amounts_passage[key] = 0
        amounts_passage[key] += 1
        sum_passage += 1

        overlay = frame.copy()
        cv2.circle(frame, (fixation[0], fixation[1]), 10, (0, 204, 0), -1)
        alpha = 0  # when set to 0 transparency of circles is 100%
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        if putNumbers:
            cv2.putText(frame, str(i + 1), (fixation[0], fixation[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                        thickness=2)
        if putLines and i > 0:
            prec_fixation = scanpath[i - 1].astype(int)
            new_color = (i * 4) % 256

            ## For arrow ->
            if prec_fixation[0] > fixation[0] or prec_fixation[1] > fixation[1]:
                left_color[left_ind] = new_color
                left_color[right_ind] = 0
                # TODO: Switch the commented lines for an arrowed line
                # cv2.arrowedLine(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]),
                #                 left_color, thickness=3, shift=0)
                cv2.line(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]),
                         left_color, thickness=3, shift=0)
            ## For arrow <-
            else:
                right_color[right_ind] = new_color
                right_color[left_ind] = 0
                # cv2.arrowedLine(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]),
                #                 right_color, thickness=3, shift=0)
                cv2.line(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]),
                         right_color, thickness=3, shift=0)

            ## Initialize new color
            if new_color == 0:
                color1 = random.randint(128, 256)
                left_color = [0, 0, 0]
                right_color = [0, 0, 0]
                left_ind = 0
                right_ind = 0
                while left_ind == right_ind:
                    left_ind = random.randint(0, 2)
                    right_ind = random.randint(0, 2)

                for index in range(1, 3):
                    if index != left_ind and index != right_ind:
                        left_color[index] = color1

                left_color[left_ind] = 128

        # if animation is required, frames are attached in a sequence
        # if not animation is required, older frames are removed
        toPlot.append(frame)
        if not animation:
            toPlot.pop(0)

    # if required, resize the frames
    if plotMaxDim:
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

    for current_key in amounts_passage:
        overlay = frame.copy()
        cv2.circle(overlay, current_key, 25 + amounts_passage.get(current_key), (0, 204, 0), -1)
        alpha = 0.4  # when set to 0 transparency of circles is 100%
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    toPlot.append(frame)


    # # draw bins
    # overlay = frame.copy()
    # for current_bin in range(HORIZONTAL_BINS):
    #     overlay = overlay.copy()
    #     fixed_y = int(current_bin*HEIGHTS[QUESTION_IDX]/HORIZONTAL_BINS)
    #     xy1 = (0, fixed_y)
    #     xy2 = (WIDTHS[QUESTION_IDX] - 1, fixed_y)
    #     black = [0, 0, 0]
    #     cv2.line(overlay, xy1, xy2, black, 3)
    # toPlot.append(overlay)
    #
    # for current_bin in range(VERTICAL_BINS):
    #     overlay = overlay.copy()
    #     fixed_x = int(current_bin*WIDTHS[QUESTION_IDX]/VERTICAL_BINS)
    #     xy1 = (fixed_x, 0)
    #     xy2 = (fixed_x, HEIGHTS[QUESTION_IDX] - 1)
    #     black = [0, 0, 0]
    #     cv2.line(overlay, xy1, xy2, black, 3)
    # toPlot.append(overlay)

    if plotMaxDim:
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

        print("Now its imshow")
    import imageio
    if SHOULD_SAVE_GIF:
        # imageio.mimsave('./scanpath.gif', toPlot[:NUMBER_OF_FRAMES_TO_SAVE])
        for repeat in range(150):
            toPlot.append(toPlot[-1])
        imageio.mimsave('./' + SUBJECT_KEY + '_' + 'question' + str(QUESTION_IDX+1) + '_' + extra_text + 'scanpath.gif', toPlot[:])
    print("saving complete")
    # calculate the entropy
    distributionX = calculate_distribution(amounts_passage, sum_passage, size[0], size[1])
    entropyOfX = calculate_entropy(distributionX)
    print("Entropy of data: " + str(entropyOfX))

    for i in range(len(toPlot)):
        cv2.imshow('Scanpath of ' + SUBJECT_KEY.split('_')[0] + ' watching ' + str(QUESTION_IDX+1),
                   toPlot[i])
        # if i == 0:
        #     milliseconds = 1
        # elif i == 1:
        #     milliseconds = scanpath[0, 3]
        # else:
        #     milliseconds = scanpath[i - 1, 3] - scanpath[i - 2, 2]
        # milliseconds *= 1000

        # cv2.waitKey(250)
        cv2.waitKey(2)
    print("Wait Key for 20 seconds")
    cv2.waitKey(1000*20)
    # cv2.waitKey(wait_time)
    #
    cv2.destroyAllWindows()

    print("Finish video")


scanpath()
