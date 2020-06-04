import os
import argparse

parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument('--down_dir', type=str, default='./db')

args = parser.parse_args()

def fetch_classname(url):
	name = url.split('/')[-1].split('.')[0]
	return name

def downloadClass(classKey, URLMap):
    # download zip file
    os.system(f"wget --no-check-certificate {URLMap[classKey]} -P .")

    # unzip zip file
    os.system(f"unzip {classKey}.zip")

    # find img directory and move to db directory
    if not os.path.exists(f'{args.down_dir}'):
        os.mkdir(f'{args.down_dir}')
    os.mkdir(f'{args.down_dir}/{classKey}')
    os.system(f"mv ./{classKey}/img/* {args.down_dir}/{classKey}/")
    
    # remove the directories after unzipping
    os.system(f'rm ./{classKey}.zip')
    os.system(f'rm -r ./{classKey}/')


# Include new URLs from the Visual Tracker dataset here.
urls = ['http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Basketball.zip',
		'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Biker.zip',
		# 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Freeman4.zip',
		# 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/BlurCar2.zip',
		# 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Bird1.zip'
	]


main_dict = {}

for url in urls:
	class_ = fetch_classname(url)
	main_dict[class_] = url

for key in main_dict.keys():
    downloadClass(key, main_dict)
