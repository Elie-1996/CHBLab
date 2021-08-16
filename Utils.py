import numpy as np
import os
from Visualization import AOI
from DataLib.GetData import Data

# TODO: Idea: It will likely prove useful to have an interface to read dataframes when you know the technology is
#  eyetracking model. Consider implementing it for improving future results.

strategy = 3


def getTime(minutes, seconds, minutes2, seconds2):
    return (minutes*60 + seconds, minutes2*60 + seconds2)


def getNextMin(minutes, seconds):
    return (minutes*60 + seconds, (minutes+1)*60 + seconds)


def getStrategy(subject, qst_idx, strategyidx):
    error = None
    if subject == '1000':
        if qst_idx == 1:
            if strategyidx == 1:
                return getTime(0, 55, 1, 53)
            elif strategyidx == 2:
                return getTime(1, 53, 2, 41)
            elif strategyidx == 3:
                return getTime(2, 41, 2, 55)
            else:
                return error
        elif qst_idx == 2:
                return error
        elif qst_idx == 3:
            if strategyidx == 1:
                return getTime(5, 15, 5, 30)
            elif strategyidx == 2:
                return getTime(5, 30, 6, 24)
            elif strategyidx == 3:
                return getTime(6, 24, 6, 55)
            else:
                return error
        elif qst_idx == 4:
            if strategyidx == 1:
                return getTime(7, 11, 7, 29)
            elif strategyidx == 2:
                return getTime(7, 29, 7, 58)
            elif strategyidx == 3:
                return getTime(7, 58, 8, 11)
            else:
                return error
        else:
            return error
    elif subject == '009':
        if qst_idx == 1:
            if strategyidx == 1:
                return getTime(1, 17, 1, 44)
            elif strategyidx == 2:
                return getTime(1, 44, 2, 45)
            elif strategyidx == 3:
                return getTime(2, 45, 5, 32)
            else:
                return error
        elif qst_idx == 2:
            return error
        elif qst_idx == 3:
            if strategyidx == 1:
                return getTime(9, 22, 9, 41)
            elif strategyidx == 2:
                return getTime(9, 41, 10, 26)
            elif strategyidx == 3:
                return getTime(10, 26, 10, 54)
            elif strategyidx == 4:
                return getTime(10, 54, 11, 59)
            else:
                return error
        elif qst_idx == 4:
            if strategyidx == 1:
                return getTime(12, 11, 13, 24)
            elif strategyidx == 2:
                return getTime(13, 24, 15, 55)
            elif strategyidx ==3:
                return getTime(15, 55, 16, 40)
            else:
                return error
        else:
            return error
    elif subject == '001':
        if qst_idx == 1:
            if strategyidx == 1:
                return getTime(5, 36, 7, 57)
            elif strategyidx == 2:
                return getTime(7, 57, 11, 12)
            else:
                return error
        elif qst_idx == 2:
            return error
        elif qst_idx == 3:
            return error
        elif qst_idx == 4:
            return error
        else:
            return error
    elif subject == '007':
        if qst_idx == 1:
            if strategyidx == 1:
                return getTime(0, 59, 2, 5)
            elif strategyidx == 2:
                return getTime(2, 5, 5, 7)
            else:
                return error
        elif qst_idx == 2:
            return error
        elif qst_idx == 3:
            return error
        elif qst_idx == 4:
            return error
        else:
            return error

    return error


input_fixations_directory = os.path.join('Subjects', 'fixations')  # CSV files
input_blinks_directory = os.path.join('Subjects', 'blinks')  # CSV files
input_pupil_directory = os.path.join('Subjects', 'pupil')  # CSV files

background_images = ['Question1.jpg', 'Question2.jpg', 'Question3.jpg', 'Question4.jpg']

subjects_dict = {
    # 'None' value means we skip the analysis of that question, and tuple (start time, end time) means we partake
    # these specific times (in seconds) to include in question
    # For example: given this data: '000': [None, (12, 43), (50, 90), None]
    # then we will skip the first question, include seconds 12 through 43 in the second
    # question, include seconds 50 through 90 in the third question and skip the fourth question
    # Note: If you would like to exclude a subject entirely -
    # simply fill them with 'None' values, example to exclude subject 9:
    # '009': [None, None, None, None]
    # ##########################################################

    # subject 001 times
    'whole_001': [getTime(5, 36, 11, 12), getTime(11, 32, 13, 8), getTime(13, 20, 16, 36), getTime(17, 2, 18, 34)],
    'min_001': [getNextMin(5, 36), getNextMin(11, 32), getNextMin(13, 20), getNextMin(17, 2)],
    'strategy_001': [getStrategy('001', 1, strategy), getStrategy('001', 2, strategy), getStrategy('001', 3, strategy), getStrategy('001', 4, strategy)],

    # subject 007 times
    'min_007':  [getNextMin(0, 59), getNextMin(6, 1), getNextMin(7, 48), getNextMin(10, 33)],
    'whole_007':  [getTime(0, 59, 5, 50), getTime(6, 1, 7, 36), getTime(7, 48, 10, 23), getTime(10, 33, 13, 44)],
    'strategy_007': [getStrategy('007', 1, strategy), getStrategy('007', 2, strategy), getStrategy('007', 3, strategy), getStrategy('007', 4, strategy)],

    # subject 009 times
    'min_009':  [getNextMin(0, 47), getNextMin(7, 36), getNextMin(9, 22), getNextMin(12, 11)],
    'whole_009':  [getTime(1, 17, 6, 51), getTime(7, 36, 9, 12), getTime(9, 22, 11, 59), getTime(12, 11, 16, 40)],
    'strategy_009':  [getStrategy('009', 1, strategy), getStrategy('009', 2, strategy), getStrategy('009', 3, strategy), getStrategy('009', 4, strategy)],

    # subject 1000 times
    'whole_1000': [getTime(0, 55, 2, 55), getTime(3, 22, 5, 2), getTime(5, 16, 6, 56), getTime(7, 11, 8, 11)],
    'min_1000': [getNextMin(0, 55), getNextMin(3, 22), getNextMin(5, 16), getNextMin(7, 11)],
    'strategy_1000': [getStrategy('1000', 1, strategy), getStrategy('1000', 2, strategy), getStrategy('1000', 3, strategy), getStrategy('1000', 4, strategy)],
}


# TODO: After Renovating Scanpath and MeanShift.
# TODO: Rename to QuestionImageWidths
WIDTHS = [1808, 2046, 1810, 1518]  # The width of image for each question (here 2046 width for Question 2)
# TODO: Rename to QuestionImageHeights
HEIGHTS = [1013, 1155, 1014, 847]  # The height of image for each question (here 1014 width for Question 3)


# TODO: Make a class describing what you are outputting, this will help encapsulate what is being output
# question selection
QUESTION_IDX = 1
img_path = os.path.join('..', 'Heatmap', background_images[QUESTION_IDX])
height = HEIGHTS[QUESTION_IDX]
width = WIDTHS[QUESTION_IDX]

# subject selection
SUBJECT_ID = '1000'
TIMESTAMPINFO = 'min'  # TODO: Turn this to a list selection
SUBJECT_KEY = TIMESTAMPINFO + '_' + SUBJECT_ID  # take the key from subjects_dict (imported above :) )
subject_path = os.path.join('..', input_fixations_directory, SUBJECT_ID + "_fixations.csv")
subject_times = subjects_dict[SUBJECT_KEY][QUESTION_IDX]


# Offset all points
#  TODO: It might be a good idea to allow any custom offsets to the point by creating a class which allows any kind of
#   alterations to the points, but it is deemed unnecessary for now. I might do it if I have the time.
OFFSET_X, OFFSET_Y = 0, 0


def get_random_array_with_range(shape, min_range, max_range):
    return np.random.rand(shape) * (max_range - min_range) + min_range


def match_fixation_to_aoi():
    """
    iterate over fixations df, check if the fixations match one of the AOI's and assign if needed
    """
    fixations_df = Data.read_only_fixation_data(get_normalized=False)
    fixations_df["AOI"] = None

    # Insert fixations to the matching AOI
    for i, row in fixations_df.iterrows():
        for aoi_num, bound in enumerate(AOI.AOI_dict["1"]):
            if bound[0][0] <= row['X'] <= bound[3][0] and bound[3][1] <= row['Y'] <= bound[0][1]:
                # print("AOI Detected")
                fixations_df['AOI'].iloc[i] = aoi_num
                break

    # for i, row in fixations_df.iterrows():
    #     if row['AOI'] is not None:
    #         print(row['AOI'])
