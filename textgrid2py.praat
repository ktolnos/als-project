# TITLE OF THE SCRIPT: get a textgrid (silences) object for use in Python

form textgrid2py.praat
text sound
text min_hz
text time_step
text silence_thresh
text min_silence_interval
text min_sound_interval
text silent_interval_label
text sound_interval_label
endform

fileName$ = sound$

Read from file: fileName$
Extract one channel... 1
Rename... original

select Sound original

To TextGrid (silences)... number(min_hz$) number(time_step$) number(silence_thresh$) number(min_silence_interval$) number(min_sound_interval$) silent_interval_label$ sound_interval_label$
Rename... original
n_sil = Get number of intervals... 1

if n_sil>0
	select TextGrid original
	int1$ = Get label of interval... 1 1
	if int1$="sounding"
		start = Get start time of interval... 1 1
	else
		# based on how textgrid (silences) works, if there is >1 interval, and the first is silent, the second must be sound (otherwise would be 1 interval) because we only define 2 labels
		start = Get start time of interval... 1 2 

	endif

	#writeInfo: start
	
	select TextGrid original
	Remove
else
	#appendInfoLine: "broken"

	select TextGrid original
	Remove
endif





	
