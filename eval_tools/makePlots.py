import sys
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import matplotlib2tikz as plt2tikz

def parseFileName(filename):
  # chop the extension off
  if (filename[-4:] == '.bag'):
    filename = filename[:-4]
  # separate into directory and filename
  index = filename.rfind('/') + 1
  directory = filename[:index]
  filename = filename[index:]
  return directory, filename


def parseTopics(topics):
  topic_list = topics.split(',')
  for i in range(len(topic_list)):
    topic_list[i] = topic_list[i].replace('/','_')
  return topic_list


def getOutputFileNames(topic_list, base_name, err_plot, extension):
  csv_file_names = []
  for topic in topic_list:
    if err_plot:
      csv_file_names.append(base_name + '_errors' + topic + extension)
    else:
      csv_file_names.append(base_name + '_aligned' + topic + extension)
  return csv_file_names

def loadData(directory, csv_file_names):
  data = []
  for file_name in csv_file_names:
    data.append(np.loadtxt(open(directory + file_name, 'rb'), delimiter=','))
  return data

def downSampleData(data):
  max_len = 500
  for i in range(0,len(data)):
    if data[i].shape[0] > max_len:
      keep_indices = np.linspace(0, data[i].shape[0] - 1, max_len);
      data[i] = data[i][keep_indices.astype(int),:]


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-filename', type=str, help='filename of the input bag')
  parser.add_argument('-topics', type=str, help='comma-separated list of topics')
  parser.add_argument('-legend', type=str, help='comma-separated list of legend labels')
  parser.add_argument('--err_plot', help='plot errors instead of states', action='store_true')

  args = parser.parse_args()

  directory, bag_filename = parseFileName(args.filename)
  topic_list = parseTopics(args.topics)
  csv_file_names = getOutputFileNames(topic_list, bag_filename, args.err_plot, '.csv')
  data = loadData(directory, csv_file_names)
  downSampleData(data)

  for log in data:
    first_time = log[0,0]
    for i in range(log.shape[0]):
      log[i,0] = log[i,0] - first_time

  fig, axes = plt.subplots(3,1)
  axes[0].set_xlabel('time (s)')
  axes[1].set_xlabel('time (s)')
  axes[2].set_xlabel('time (s)')

  if args.err_plot:
    axes[0].set_ylabel('x velocity error (m/s)')
    axes[1].set_ylabel('y velocity error (m/s)')
    axes[2].set_ylabel('z velocity error (m/s)')
  else:
    axes[0].set_ylabel('x velocity (m/s)')
    axes[1].set_ylabel('y velocity (m/s)')
    axes[2].set_ylabel('z velocity (m/s)')

  start = 0
  if args.err_plot:
    start = 1

  for i in range(start,len(data)):
    axes[0].plot(data[i][:,0], data[i][:,1])
    axes[1].plot(data[i][:,0], data[i][:,2])
    axes[2].plot(data[i][:,0], data[i][:,3])

  axes[0].legend(args.legend.split(','))

  response = raw_input('save to tikz? (y/n)\n')

  if response == 'y':
    tex_file_name = directory + bag_filename
    if args.err_plot:
      tex_file_name = tex_file_name + '_err_plot.tex'
    else:
      tex_file_name = tex_file_name + '_state_plot.tex'
    plt2tikz.save(tex_file_name)
  else:
    plt.show()