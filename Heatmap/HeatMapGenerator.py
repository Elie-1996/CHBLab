import math
import os
from PIL import Image
from heatmappy import Heatmapper
from heatmappy.video import VideoHeatmapper
import pandas as pd
import cv2
from Utils import background_images, WIDTHS, HEIGHTS, subjects_dict, input_fixations_directory, strategy


QUESTION_IDX = 2
# extra_text = 'min_'
extra_text = 'whole_'
# extra_text = 'strategy' + str(strategy) + '_'  # don't forget to add strategy index/tag!
# extra_text = 'IGNORE_THIS_RESULT' + str(strategy) + '_'
SUBJECT_KEY = '001'  # take the key from subjects_dict (imported above :) )
NAMEOFFILE = SUBJECT_KEY + '_Question' + str(QUESTION_IDX + 1) + '_' + extra_text +'.png'
HM_INTERVAL = 1
filename = background_images[QUESTION_IDX]
CRED = '\33[32m'
CEND = '\033[0m'
# OFFSET_X, OFFSET_Y = -250, -46
OFFSET_X, OFFSET_Y = 0, 0

def produce_interval_heatmap():
    ##  For image
    for subject in subjects_dict.keys():
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        img = Image.open(filename)
        heatmapper = Heatmapper()
        img_list = []
        input_points = os.path.join('..', input_fixations_directory, subject + '_fixations.csv')
        current_subject_times = subjects_dict[subject]
        current_subject_time = current_subject_times[QUESTION_IDX]
        if current_subject_time is None:
            continue
        df = pd.read_csv(input_points)
        interval = current_subject_time[1] - current_subject_time[0]    # Number of seconds
        num_rows = len(df)
        normalize_time = df['start_timestamp'].iloc[0]
        df['start_timestamp'] -= normalize_time
        idx = 0
        while True:
            if idx >= num_rows or df['start_timestamp'].iloc[idx] > current_subject_time[1]:
                break
            if df['on_surf'].iloc[idx] and current_subject_time[0] <= df['start_timestamp'].iloc[idx]:
                start_time = df['start_timestamp'].iloc[idx]
                current_points = []
                while True:
                    if idx < num_rows and df['start_timestamp'].iloc[idx] - start_time <= interval:
                        if df['on_surf'].iloc[idx] and current_subject_time[0] <= df['start_timestamp'].iloc[idx] <= current_subject_time[1]:
                            finalx, finaly = df['norm_pos_x'].iloc[idx] * size[0] + OFFSET_X, (1 - df['norm_pos_y'].iloc[idx])*size[1] + OFFSET_Y
                            if finalx >= 0 and finalx <= size[0] and finaly >= 0 and finaly <= size[1]:
                                current_points.append((df['norm_pos_x'].iloc[idx] * size[0] + OFFSET_X, (1 - df['norm_pos_y'].iloc[idx])*size[1] + OFFSET_Y))
                        idx += 1
                    else:
                        break
                img_list.append(heatmapper.heatmap_on_img(current_points, img))
            idx += 1

        print("Saving")
        img_list[0].save(NAMEOFFILE)
        # Uncomment for video
        # out = cv2.VideoWriter(f'Interval-HeatMap-Question-{QUESTION_IDX}-Subject-'+subject.split('_')[0]+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, size)
        #
        # for i in range(len(img_list)):
        #     img_list[i].save('heatmap.png')
        #     for j in range(5):
        #         img = cv2.imread('heatmap.png')
        #     out.write(img)
        # out.release()


if __name__ == '__main__':
    produce_interval_heatmap()
