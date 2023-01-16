import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='train.csv', type=str, help='file path of data. (.csv)')
    parser.add_argument('--width', default=195, type=int, help='size of x-axis [mm]')
    parser.add_argument('--height', default=113, type=int, help='size of y-axis [mm]')
    parser.add_argument('--pixels_w', default=1400, type=int, help='number of pixels in x-axis')
    parser.add_argument('--pixels_h', default=1400, type=int, help='number of pixels in y-axis')
    parser.add_argument('--distance', default=450, type=int, help='distance between eye and observing plane [mm]')
    parser.add_argument('--d_max', default=3.0, type=float, help='maximum dispersion threshold [degree]')
    parser.add_argument('--f_min', default=100.0, type=float, help='minimum fixation duration threshold [ms] (100~200ms is recommended)')
    parser.add_argument('--sampling_rate', default=1000, type=float, help='sampling rate of collected data [Hz]')
    parser.add_argument('--datanum', default=None, type=int, help='use only datanum data')
    parser.add_argument('--visualize', default=None, type=int, help='visualize result of i-th data')
    # parser.add_argument('--distance', default=1000, type=float, help='distance between eyes and ovserving plane [mm]')

    return parser.parse_args()



def load_data(filename, distance, width, height, pixels_w, pixels_h, n):
    with open(filename, 'r') as file:
        f_list = file.readlines()
    
    # unit size of each pixels
    w = width / pixels_w
    h = height / pixels_h
    
    if n == None:
        n = len(f_list) - 1
    dataset = []
    for f in tqdm(f_list[:n], desc='loading...'):
        f = f.split(',') # convert initial f='s26,false,-739.34,...' into f=['s26','false','-739.34',...]

        data = {}
        data['sid'] = f[0]
        data['known'] = f[1]
        data['xy'] = []
        for i in range(int((len(f)-2)/2)): # number of (x,y) data
            x = float(f[i*2+2]) * w
            y = float(f[i*2+3]) * h
            x_angle = np.arctan(x/distance)
            y_angle = np.arctan(y/distance)
            data['xy'].append((x_angle, y_angle))
            # data['xy'].append((float(f[i*2+2]), float(f[i*2+3])))
        data['xy'] = np.array(data['xy'])
        dataset.append(data)

    return dataset



def calc_dispersion(data):
    return (np.max(data[:, 0]) - np.min(data[:, 0])) + (np.max(data[:, 1]) - np.min(data[:, 1]))



def calc_centroid(data):
    return (np.mean(data[:, 0]), np.mean(data[:, 1]))



# data: collection of position (Gaze point X, Gaze point Y)
# d_max: maximum dispersion threshold [degree]
# f_min: minimum fixation duration threshold [ms]
def dispersion_threshold_algorithm(data, d_max, fp_min):
    fixation_groups = []
    i = 0
    while i < len(data)-fp_min:
        start = i
        end = i + fp_min - 1
        if calc_dispersion(data[start:end]) <= d_max:
            while end < len(data)-1:
                end += 1
                if calc_dispersion(data[start:end]) > d_max:
                    end -= 1 # last point isn't included in fixation group
                    break
            fixation_groups.append(((start, end), calc_centroid(data[start:end])))
            i += len(range(start, end)) + 1 # remove window points from points
        else:
            i += 1 # remove first point from points
        
    return fixation_groups



if __name__ == '__main__':
    args = get_args()
    filename = args.filename
    width = args.width
    height = args.height
    pixels_w = args.pixels_w
    pixels_h = args.pixels_h
    distance = args.distance
    d_max = args.d_max * np.pi/180 # convert degree into radian
    f_min = args.f_min
    sampling_rate = args.sampling_rate
    datanum = args.datanum
    visualize_i = args.visualize
    print('max dispersion: %.1f [degree]' %(d_max*180/np.pi))
    print('min fixation duration: %.1f [ms]' %f_min)
    print('sampling rate: %.1f [Hz]' %sampling_rate)
    fp_min = int(f_min / (1000 / sampling_rate)) # minimum number of points covered by window
    
    dataset = load_data(filename, distance, width, height, pixels_w, pixels_h, datanum)
    
    results = []
    for data in dataset:
        result = {}
        result['sid'] = data['sid']
        result['known'] = data['known']
        result['fixation_groups'] = dispersion_threshold_algorithm(data['xy'], d_max, fp_min) # get collections of fixation duration and centroid
        results.append(result)


    if visualize_i != None:
        fixation_groups = results[visualize_i]['fixation_groups']
        fixations = [] # indexis of points detected as fixation
        for duration, centroid in fixation_groups:
            fixations.extend(np.arange(duration[0], duration[1]))
        print('number of fixation groups:', len(fixation_groups))
        print('number of fixation points:', len(fixations))

        # make dataframe to visualize I-DT result
        _result = []
        for xy in dataset[visualize_i]['xy']:
            _r = [xy[0], xy[1], None]
            _result.append(_r)
        df_result = pd.DataFrame(_result, columns=['x', 'y', 'Eye movement type'])
        for i in range(len(df_result)):
            if i in fixations:
                df_result.at[i, 'Eye movement type'] = 'Fixation'
            else:
                df_result.at[i, 'Eye movement type'] = 'Saccade'
        

        fig, ax = plt.subplots(ncols=1, figsize=(7,7))
        idt_colors = ['green' if x == 'Fixation' else 'red' for x in df_result['Eye movement type']]
        ax.scatter(df_result['x'], df_result['y'], color=idt_colors, s=3)
        ax.set_title('%s %s' %(results[visualize_i]['sid'], results[visualize_i]['known']), fontdict={'fontsize':20})
        plt.show()



'''
distance between eye and observing plane
d = 450 mm

observing plane size
1400 x 1400 pixels
height = 113 mm
width = 195 mm

each pixel size
h = 195 / 1400 mm
w = 113 / 1400 mm

normalized coordinate position
_x = x_raw * w
_y = y_raw * h

angle
x = arctan(_x/d) degree/radian
y = arctan(_y/d) degree/radian
'''