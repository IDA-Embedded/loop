
audio_info:
	mediainfo audio_1.wav
	
audio_split: 
	mkdir -p obj ;\
	ffmpeg -i audio_1.wav -f segment -segment_time 60 -c copy obj/out%03d.wav
