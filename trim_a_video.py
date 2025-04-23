import sys
import getopt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def get_input(argv, unit='second'):
    opts, args = getopt.getopt(argv,"f:u:",["filename=", "unit=", "ti=", "tf="])
    for opt, arg in opts:
        if opt in ("-f", "--filename"):
            filename = arg
        elif opt in ("--ti"):
            ti = int(arg)
        elif opt in ("--tf"):
            tf = int(arg)
        elif opt in ("-u", "--unit"):
            unit = arg
        else:
            print('Error: Unknown option {}'.format(opt))
            sys.exit()
    return filename, ti, tf, unit


def trim_video(filename, ti, tf, unit='second', filename_trimmed=None):
    if unit in ['m', 'min', 'minut', 'minute', 'minutes']:
        ti = ti * 60
        tf = tf * 60
    if filename_trimmed==None:
        filename_trimmed = filename[:-4] # Remove .mp4
        filename_trimmed += '_trimmed_from_' + str(ti) + '_s_to_' + str(tf) + '_s'
        filename_trimmed += '.mp4'
    
    ffmpeg_extract_subclip(filename, ti, tf, targetname=filename_trimmed)


if __name__ == '__main__':
    
    filename, ti, tf, unit = get_input(sys.argv[1:])
    
    if unit in ['m', 'min', 'minut', 'minute', 'minutes']:
        ti = ti * 60
        tf = tf * 60
    
    filename_trimmed = filename[:-4] # Remove .mp4
    filename_trimmed += '_trimmed_from_' + str(ti) + '_s_to_' + str(tf) + '_s'
    filename_trimmed += '.mp4'
    
    ffmpeg_extract_subclip(filename, ti, tf, targetname=filename_trimmed)
    # /!\ ti and tf must be in seconds.