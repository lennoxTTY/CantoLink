from multiprocessing import Process
from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, current_app, send_file, url_for
import uuid, math, datetime, ffmpeg,pysrt,os,shutil,auditok,timeit,threading,torch
import azure.cognitiveservices.speech as speechsdk
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip,TextClip,CompositeVideoClip,AudioFileClip
from transformers import BertTokenizer, BartForConditionalGeneration

speech_key= "719ce10041a44d2b8f4b4ebbd1744030"
location = "eastasia"
src_folder = r"D:\HKMU Assignment\FYP\DemoPrototype\output_Preview.mp4"        # change this source path of embed video output 
dst_folder = r"D:\HKMU Assignment\FYP\DemoPrototype\static\output_Preview.mp4"     # change this destiny path to place output in static folder
tokenizer = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')
model = BartForConditionalGeneration.from_pretrained("D:/HKMU Assignment/FYP/DemoPrototype/model/")  # change this to model folder path
device = torch.device("cuda:0")
model = model.to(device)

def create_app():
    app = Flask(__name__)
    Bootstrap(app)
    return app
app=create_app()

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=location)

inputFilename=''

def no_file_error(msg):
    return render_template('index.html',msg=msg)

def split_audio(sub_times,th):
    audio_regions = auditok.split(
        "output.wav",
        min_dur=1,     # minimum duration of a valid audio event in seconds
        max_dur=10,       # maximum duration of an event
        max_silence=0.1, # maximum duration of tolerated continuous silence within an event
        energy_threshold=th # threshold of detection
    )
    for index, clip in enumerate(audio_regions):
        start = math.modf(float(f"{clip.meta.start:.3f}"))
        end = math.modf(float(f"{clip.meta.end:.3f}"))
        times=[start,end]
        sub_times[index] = times
        clip.save(f"region_{index}.wav")

def recog(index,list):
    filename=f'region_{index}.wav'
    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, language="zh-HK", audio_config=audio_config)
    speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "LogfilePathAndName")
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:  
        print("Recognized: {}".format(result.text))
        list[index] = result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:  
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:  
        cancellation_details = result.cancellation_details  
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))  
        if cancellation_details.reason == speechsdk.CancellationReason.Error:  
            print("Error details: {}".format(cancellation_details.error_details))

def editScript(line,ignore):
    ignore = ignore+",，,？,。"
    ignore = ignore.split(',')
    for word in ignore:
        if word in line:
            line = line.replace(word,'')
    return line

    
@app.route("/")
def index():
    return render_template('index.html',msg="字幕內容預覽")

@app.route("/go_back")
def go_back():
    return redirect(url_for('index')+"#tool-page-scroll")

@app.route("/takeFile",methods=['GET','POST'])
def takeFile():
    global inputFilename
    if request.method=='POST':
        totalTimeS = timeit.default_timer()
        sub_times={}
        # getting threshold slider
        try:
            threshold=int(request.form["th"])
        except:
            threshold = 59

        # getting video file
        try:
            fileInput=request.files['videofile']
            filename = secure_filename(fileInput.filename)
            inputFilename=filename
            fileInput.save(filename)
        except:
            no_file_error("無法讀取檔案")

        # extract full audio
        (ffmpeg.input(filename)
        .output("output.wav")
        .run(overwrite_output=True))

        # extract preview video part
        (ffmpeg.input(filename, ss="00:00:00", to="00:00:10")
        .output("outputPreview.mp4", vcodec="libx264", crf="28")
        .run(overwrite_output=True))

        # splitting audio by line
        split_audio(sub_times,th=threshold)
        sub_dict = {}

        # audio regcognization
        threadList = [threading.Thread(target=recog, args=[i,sub_dict]) for i in range(len(sub_times))]
        for t in threadList: t.start()
        for t in threadList: t.join()

        # ignoring words from list
        try:
            ignore = request.form["csignore_list"]
        except:
            ignore = ""
        try:
            ignore += request.form["cmignore_list"]
        except:
            ignore += ""
        for k,t in sub_dict.items():
            sub_dict[k]=editScript(t,ignore)

        # editing SRT file
        editSRT(sub_times,sub_dict)

        Outputfile = pysrt.open("output.srt", encoding="UTF-8")
        os.remove("output.wav")
        try:
            loadVideo("outputPreview.mp4","output_Preview.mp4",fontsize=36,font="Microsoft-JhengHei-Light-&-Microsoft-JhengHei-UI-Light",color="white",bg_color="gray",isPreview=True)
            shutil.move(src_folder,dst_folder)
        except:
            print("Could not generate Preview video")
        ##create and edit .SRT with timestamp and transcribtion
        totalTimeE = timeit.default_timer()
        time = totalTimeE-totalTimeS
        time = round(time,2)
        return render_template('processed.html',len=len(Outputfile),subtitles=Outputfile,time=f"處理時間：{time}秒")
    else:
        no_file_error("無法讀取檔案")

def translator(index,original_text,list):
    input_ids = tokenizer([original_text], return_tensors="pt", max_length=200, truncation=True)["input_ids"].to(device)
    result = model.generate(input_ids, max_new_tokens=100)
    outputs = tokenizer.batch_decode(result, skip_special_tokens=True)
    list[index]=outputs[0].replace(" ", "")
    print(f"{index} line is done!")

def editSRT(sub_times,sub_dict):
    output = open("output.srt", "w+", encoding="utf-8")
    translate_req = {}
    # text part
    threadList = [threading.Thread(target=translator, args=[i,sub_dict[i],translate_req]) for i in range(len(sub_dict))]
    for t in threadList: t.start()
    for t in threadList: t.join()
    if len(translate_req) == len(sub_dict):
        for j in range(len(sub_times)):

            # indexing each line
            output.write(str(j+1)+"\n")
            start = sub_times[j][0]
            end = sub_times[j][1]

            # converting times to srt format
            # d/f is before/after fractional
            startd = start[1]
            startf = str(round(start[0], 3)).split(".")[1]
            if len(startf) == 1:
                startf += "00"
            endd = end[1]
            endf = str(round(end[0], 3)).split(".")[1]
            if len(endf) == 1:
                endf += "00"
            startdt = str(datetime.timedelta(seconds=startd))
            enddt = str(datetime.timedelta(seconds=endd))
            startTime = startdt + "," + startf
            endTime = enddt + "," + endf

            # writing start and end time of subtitle
            output.write(startTime+" --> "+endTime+"\n")

            # write text
            output.write(translate_req[j]+"\n\n")
            os.remove(f"region_{j}.wav")
    print(f"successfully written srt")

@app.route('/download', methods=['GET', 'POST'])
def download_file():
    path="output.srt"
    return send_file(path,as_attachment=True)

@app.route('/preview',methods=['GET', 'POST'])
def preview():
    try:
        previewTimeS = timeit.default_timer()
        fontsize = int(request.args.get('fontsize'))
        font = request.args.get('font')
        color = request.args.get('color')
        bg_color = request.args.get('bg_color')
        previewThread = Process(target=loadVideo,args=("outputPreview.mp4","output_Preview.mp4",fontsize,font,color,bg_color,True))
        previewThread.start()
        # generating full video for download in the background
        fullvideoThread = Process(target=loadVideo,args=(inputFilename,"output_subtitled.mp4",fontsize,font,color,bg_color,False))
        fullvideoThread.start()

        previewThread.join()
        shutil.move(src_folder,dst_folder)
        #return render_template('takeFile.html')
        previewTimeE = timeit.default_timer()
        time=previewTimeE-previewTimeS
        return jsonify({"stat":"success","time":time})
    except:
        return jsonify(({"stat":"Failed to generate preview"}))

@app.route('/previewDownload', methods=['GET', 'POST'])
def previewDownload():
    path="output_subtitled.mp4"
    return send_file(path,as_attachment=True)

def time_to_seconds(time_obj):
    return time_obj.hours*3600+time_obj.minutes*60+time_obj.seconds+time_obj.milliseconds/1000

def burnInSubtitle(sub,videosize=None,fontsize=24,font="Microsoft-JhengHei-Light-&-Microsoft-JhengHei-UI-Light",color='white',bg_color="black",debug=False):
    sub_clips=[]
    for subtitle in sub:
        start_time=time_to_seconds(subtitle.start)
        end_time=time_to_seconds(subtitle.end)
        duration=end_time-start_time
        video_width, video_height = videosize
        text_clip=TextClip(subtitle.text,fontsize=fontsize,font=font,color=color,bg_color=bg_color,size=(video_width*3/4,None),method='caption').set_start(start_time).set_duration(duration)
        subtitle_x_position='center'
        subtitle_y_position=video_height*4/5
        text_position=(subtitle_x_position,subtitle_y_position)
        sub_clips.append(text_clip.set_position(text_position))
    return sub_clips

def loadVideo(video,outputName,fontsize,font,color,bg_color,isPreview):
    video=VideoFileClip(video)
    subtitle=pysrt.open('output.srt')
    if isPreview == True:
        # get subtitle in the first 15 seconds
        try:
            LastSubtitle = subtitle.slice(starts_before={'minutes':0,'seconds':10},ends_after={'minutes':0,'seconds':10})[0]
            LastSubtitle.end.seconds = 10
        except:
            pass
        subtitles = subtitle.slice(ends_before={'minutes':0,'seconds':10}) # get anything before 15s
        subtitle = []
        for sub in subtitles:
            subtitle.append(sub)
        try:
            subtitle.append(LastSubtitle)
        except:
            pass
    else:
        pass
    output_video=outputName
    sub_clips=burnInSubtitle(subtitle,videosize=video.size,fontsize=fontsize,font=font,color=color,bg_color=bg_color)
    final_video=CompositeVideoClip([video]+sub_clips)
    final_video.write_videofile(output_video)

if __name__=="__main__":
    app.run(debug=True)