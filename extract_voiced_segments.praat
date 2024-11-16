# TITLE OF THE SCRIPT: Extract voiced parts of sound file - adapted from Maryn, Corthals & Barsties (Acoustic Voice Quality Index 03.01)

form extract_voiced_segments.praat
text main_dir
text sound
text silenceStart
text silenceEnd
endform

dir_recordings$ = main_dir$
dir_onlyvoiced$ = main_dir$ + "/voiced"
fileName$ = sound$

Read from file: dir_recordings$ + "/" + fileName$
Extract one channel... 1
Rename... cs

# --------------------------------------------------------------------------------------------
# PART 0:
# HIGH-PASS FILTERING OF THE SOUND FILES.
# --------------------------------------------------------------------------------------------
select Sound cs
Filter (stop Hann band)... 0 34 0.1
Rename... cs2
# --------------------------------------------------------------------------------------------
# PART 1:
# DETECTION, EXTRACTION AND CONCATENATION OF
# THE VOICED SEGMENTS IN THE RECORDING
# OF CONTINUOUS SPEECH.
# --------------------------------------------------------------------------------------------
select Sound cs2
Copy... original
select Sound original
silenceStart = number(silenceStart$)
silenceEnd = number(silenceEnd$)
maxPAsilence = Get root-mean-square... silenceStart silenceEnd
appendInfoLine: maxPAsilence

select Sound original
samplingRate = Get sampling frequency
intermediateSamples = Get sampling period
Create Sound... onlyVoice 0 0.001 'samplingRate' 0 
select Sound original
To TextGrid (silences)... 50 0.003 -25 0.1 0.1 silence sounding
n_sil = Count intervals where... 1 "is equal to" silence

if n_sil!=0
	#select Sound original
	#plus TextGrid original
	#Extract intervals where... 1 no "contains" silence
	#Concatenate
	#Save as WAV file: dir_onlyvoiced$ + "/" + fileName$ + "silence.wav"
	select Sound original
	plus TextGrid original
	Extract intervals where... 1 no "contains" sounding
	Concatenate
	select Sound chain
	Rename... onlyLoud
	globalPower = Get power in air
	maxPA = Get root-mean-square... 0 0
	appendInfoLine: "\n"
	writeInfo: maxPA
	#Save as WAV file: dir_onlyvoiced$ + "/" + fileName$ + "onlyLoud.wav"
	select TextGrid original
	Remove
else
	appendInfoLine: n_sil
	select Sound original
	Rename... onlyLoud
	globalPower = Get power in air
	maxPA = Get root-mean-square... 0 0
	appendInfoLine: "\n"
	writeInfo: maxPA
	# Save as WAV file: dir_onlyvoiced$ + "/" + "onlyLoud.wav"
	select TextGrid original
	Remove
endif

select Sound onlyLoud
signalEnd = Get end time
windowBorderLeft = Get start time
windowWidth = 0.03
windowBorderRight = windowBorderLeft + windowWidth
globalPower = Get power in air
voicelessThreshold = globalPower*(30/100)

select Sound onlyLoud
extremeRight = signalEnd - windowWidth
while windowBorderRight < extremeRight
	Extract part... 'windowBorderLeft' 'windowBorderRight' Rectangular 1.0 no
	select Sound onlyLoud_part
	partialPower = Get power in air
	if partialPower > voicelessThreshold
		call checkZeros 0
		if (zeroCrossingRate <> undefined) and (zeroCrossingRate < 3000)
			select Sound onlyVoice
			plus Sound onlyLoud_part
			Concatenate
			Rename... onlyVoiceNew
			select Sound onlyVoice
			Remove
			select Sound onlyVoiceNew
			Rename... onlyVoice
		endif
	endif
	select Sound onlyLoud_part
	Remove
	windowBorderLeft = windowBorderLeft + 0.03
	windowBorderRight = windowBorderLeft + 0.03
	select Sound onlyLoud
endwhile
select Sound onlyVoice

procedure checkZeros zeroCrossingRate

	start = 0.0025
	startZero = Get nearest zero crossing... 'start'
	findStart = startZero
	findStartZeroPlusOne = startZero + intermediateSamples
	startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
	zeroCrossings = 0
	strips = 0

	while (findStart < 0.0275) and (findStart <> undefined)
		while startZeroPlusOne = findStart
			findStartZeroPlusOne = findStartZeroPlusOne + intermediateSamples
			startZeroPlusOne = Get nearest zero crossing... 'findStartZeroPlusOne'
		endwhile
		afstand = startZeroPlusOne - startZero
		strips = strips +1
		zeroCrossings = zeroCrossings +1
		findStart = startZeroPlusOne
	endwhile
	zeroCrossingRate = zeroCrossings/afstand
endproc


# Save sound file with only voiced segments
select Sound onlyVoice
fileName$ = fileName$ - ".wav" + "_OnlyVoiced"
Rename... 'fileName$'
Save as WAV file: dir_onlyvoiced$ + "/" + fileName$ + ".wav"
	

# Remove intermediate objects
select all
Remove
Erase all




	
