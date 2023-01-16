import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='train.csv', type=str, help='file path of data. (.csv)')
    parser.add_argument('--save', default='save/save', type=str, help='file path foe save. (.csv)')
    parser.add_argument('--width', default=195, type=int, help='size of x-axis [mm]')
    parser.add_argument('--height', default=113, type=int, help='size of y-axis [mm]')
    parser.add_argument('--pixels_w', default=1400, type=int, help='number of pixels in x-axis')
    parser.add_argument('--pixels_h', default=1400, type=int, help='number of pixels in y-axis')
    parser.add_argument('--distance', default=450, type=int, help='distance between eye and observing plane [mm]')
    parser.add_argument('--d_max', default=10.0, type=float, help='maximum dispersion threshold [degree]')
    parser.add_argument('--f_min', default=100.0, type=float, help='minimum fixation duration threshold [ms] (100~200ms is recommended)')
    parser.add_argument('--sampling_rate', default=1000, type=float, help='sampling rate of collected data [Hz]')
    parser.add_argument('--datanum', default=None, type=int, help='use only datanum data')
    # parser.add_argument('--visualize', default=None, type=int, help='visualize result of i-th data')

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
    for f in tqdm(f_list[:n], desc='loading data'):
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
        data['xy'] = np.array(data['xy'])
        dataset.append(data)

    return dataset



def calc_dispersion(data):
    return (np.max(data[:, 0]) - np.min(data[:, 0])) + (np.max(data[:, 1]) - np.min(data[:, 1]))



def calc_centroid(data):
    return (np.mean(data[:, 0]), np.mean(data[:, 1]))



def calc_duration(durations, hz):
    return np.sum([(end-start)*(1/hz) for start,end in durations])+1 # sec measurement, not ms



def calc_amplitude(points):
    x_sub = np.max(points[:, 0]) - np.min(points[:, 0])
    y_sub = np.max(points[:, 1]) - np.min(points[:, 1])
    amp = np.sqrt(x_sub*x_sub + y_sub*y_sub)
    return amp



# data: collection of position (Gaze point X, Gaze point Y)
# d_max: maximum dispersion threshold [degree]
# f_min: minimum fixation duration threshold [ms]
def dispersion_threshold_algorithm(data, d_max, fp_min):
    fixation_durations = []
    fixation_centroids = []
    saccade_durations = []
    saccade_amplitudes = []

    saccades = []
    i = 0
    while i < len(data)-fp_min:
        start = i
        end = i + fp_min - 1
        if calc_dispersion(data[start:end]) <= d_max:
            if len(saccades) > 0:
                saccade_durations.append((saccades[0][0], saccades[-1][0]))
                saccade_amplitudes.append(calc_amplitude(np.array([saccade[1] for saccade in saccades])))
                saccades = []

            while end < len(data)-1:
                end += 1
                if calc_dispersion(data[start:end]) > d_max:
                    end -= 1 # last point isn't included in fixation group
                    break
            fixation_durations.append((start, end-1))
            fixation_centroids.append(calc_centroid(data[start:end]))
            i += len(range(start, end)) # remove window points from points

        else:
            saccades.append((i, data[i]))
            i += 1 # remove first point from points
        
    return fixation_durations, fixation_centroids, saccade_durations, saccade_amplitudes



if __name__ == '__main__':
    args = get_args()
    filename = args.filename
    save = args.save
    width = args.width
    height = args.height
    pixels_w = args.pixels_w
    pixels_h = args.pixels_h
    distance = args.distance
    d_max = args.d_max * np.pi/180 # convert degree into radian
    f_min = args.f_min
    sampling_rate = args.sampling_rate
    datanum = args.datanum
    # visualize_i = args.visualize
    print('max dispersion: %.1f [degree]' %(d_max*180/np.pi))
    print('min fixation duration: %.1f [ms]' %f_min)
    print('sampling rate: %.1f [Hz]' %sampling_rate)
    fp_min = int(f_min / (1000 / sampling_rate)) # minimum number of points covered by window

    dataset = load_data(filename, distance, width, height, pixels_w, pixels_h, datanum)
    selected_sid = ['s2','s4','s8','s10','s16','s22','s23','s30']
    dataset_selected = [data for data in dataset if data['sid'] in selected_sid] # extract assigned subjects from dataset

    results = []
    for data in tqdm(dataset_selected, desc='computing I-DT'):
        result = {}
        result['sid'] = data['sid']
        result['known'] = data['known']
        result['f_durations'], result['f_centroids'], result['s_durations'], result['s_amplitudes'] \
            = dispersion_threshold_algorithm(data['xy'], d_max, fp_min)
        results.append(result)
    
    '''
    NF: number of fixation
    FD: fixation duration
    SA: saccade amplitude
    '''
    NFs, FDs, SAs = {}, {}, {}
    # SDs = {}
    for sid in selected_sid:
        NFs[sid] = {'true':[], 'false':[]}
        FDs[sid] = {'true':[], 'false':[]}
        SAs[sid] = {'true':[], 'false':[]}
        # SDs[sid] = {'true':[], 'false':[]}
    for result in results:
        sid, known = result['sid'], result['known']
        f_durations = result['f_durations']
        s_durations = result['s_durations']
        s_amplitudes = result['s_amplitudes']
        NFs[sid][known].append(len(f_durations))
        FDs[sid][known].append(calc_duration(f_durations, sampling_rate))
        if len(s_amplitudes) > 0:
            SAs[sid][known].append(np.mean(s_amplitudes))
        else:
            SAs[sid][known].append(0)
        # SDs[sid][known].append(calc_duration(s_durations, sampling_rate))
    

    ''' subject_id
        MFD_true       [sec]
        MFD_SD_true    [sec]
        MFD_false      [sec]
        MFD_SD_false   [sec]
        MSA_true       [deg]
        MSA_SD_true    [deg]
        MSA_false      [deg]
        MSA_SD_false   [deg]
        MFD_overall    [sec]
        MFD_overall_SD [sec]
        MSA_overall    [deg]
        MSA_overall_SD [deg]
    '''
    MNF, MFD, SFD, MSA, SSA, OMNF, OMFD, OSFD, OMSA, OSSA = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for sid in selected_sid:
        MNF[sid], MFD[sid], SFD[sid], MSA[sid], SSA[sid] = {}, {}, {}, {}, {}
    for sid in selected_sid:
        for known in ['true', 'false']:
            MNF[sid][known] = np.mean(NFs[sid][known])
            MFD[sid][known] = np.mean(FDs[sid][known])
            SFD[sid][known] = np.std(FDs[sid][known])
            MSA[sid][known] = np.mean(SAs[sid][known]) * 180/np.pi
            SSA[sid][known] = np.std(SAs[sid][known]) * 180/np.pi
            print('%3s-%5s: MNF=%6.3f, MFD=%6.3f, SFD=%6.3f, MSA=%6.3f, SSA=%6.3f'\
                %(sid, known, MNF[sid][known], MFD[sid][known], SFD[sid][known], MSA[sid][known], SSA[sid][known]))
        OMNF[sid] = np.mean((MNF[sid]['true'],MNF[sid]['false']))
        OMFD[sid] = np.mean((MFD[sid]['true'],MFD[sid]['false']))
        OSFD[sid] = np.mean((SFD[sid]['true'],SFD[sid]['false']))
        OMSA[sid] = np.mean((MSA[sid]['true'],MSA[sid]['false']))
        OSSA[sid] = np.mean((SSA[sid]['true'],SSA[sid]['false']))
        print('%3s-orval: MNF=%6.3f, MFD=%6.3f, SFD=%6.3f, MSA=%6.3f, SSA=%6.3f\n'\
            %(sid, OMNF[sid], OMFD[sid], OSFD[sid], OMSA[sid], OSSA[sid]))

    # For csv file 
    header = ['subject_id', 'MFD_true', 'MFD_SD_true', 'MFD_false', 'MFD_SD_false', 'MSA_true', 'MSA_SD_true', 'MSA_false',
    'MSA_SD_false', 'MFD_overall', 'MFD_overall_SD', 'MSA_overall', 'MSA_overall_SD']
    size = [len(selected_sid), len(header)]
    data = np.empty(size,dtype=object)
    i = 0
    for sid in selected_sid:
            data[i,:] = ([sid, MFD[sid]['true'], SFD[sid]['true'], MFD[sid]['false'], SFD[sid]['false'], MSA[sid]['true'], SSA[sid]['true'],
                MSA[sid]['false'], SSA[sid]['false'], OMFD[sid], OSFD[sid], OMSA[sid], OSSA[sid]])
            i = i+1

    with open('dispersion_100_results.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

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