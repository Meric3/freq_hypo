from datetime import timezone, timedelta, datetime

from pathlib import Path
import logging
import os
from tensorboardX import SummaryWriter




def make_log(opt):
    tz = timezone(timedelta(hours=9))
    now = datetime.now(tz)
    mon = format(now.month, '02')
    day = format(now.day, '02')
    h = format(now.hour, '02')
    m = format(now.minute, '02')
    s = format(now.second, '02')
    date = mon+day
    now = h+m+s
    print("Time {}, {}".format(date, now))

    if opt['local_save'] == True:
        log_dir_path = Path(opt['log_base_path'], opt['project_name'])
        log_dir_path.mkdir(parents=True, exist_ok=True)

        log_dir_path = Path(log_dir_path, date)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        tf_path = Path(log_dir_path, "tf")
        tf_path.mkdir(parents=True, exist_ok=True)

        log_dir_path = Path(log_dir_path, opt['exp'] +'_'+now)
        log_dir_path.mkdir(parents=True, exist_ok=True)    

        tf_path = Path(tf_path, now)
        tf_path.mkdir(parents=True, exist_ok=True)

        log_path = str(log_dir_path) + "/" +str(opt['pred_lag'])\
                    + str(opt['multi']) + str(opt['invasive']) + ".txt"
        
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        fileHandler = logging.FileHandler(log_path)
        fileHandler.setFormatter(formatter)
        log.addHandler(fileHandler)
        log.info("path {} ".format(log_dir_path))
        
        
    log.info("-"*99)
    log.info(now)
    log.info("-"*99)
    for name in opt:
        log.info("{} : {}".format(name, opt[name]))    

    project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

    cp_src = "cp -r " + project_path + " "+str(log_dir_path) + "/"
    print("cp_src is {}".format(cp_src))
    os.system(cp_src)

    summary = SummaryWriter(tf_path)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    
    return log, log_dir_path, summary
