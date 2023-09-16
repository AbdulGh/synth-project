#ifndef CONFIG_H
#define CONFIG_H

constexpr int sampleRate = 24000;
constexpr int duration = 4;
constexpr int framesToGenerate = sampleRate * duration;

enum parameters {
	SINECAT,
	SAWCAT,
	SQUARECAT,
	PITCH,
	CUTOFF,
	RESONANCE,
	ATTACKTIME,
	DECAYTIME,
	SUSTAINLEVEL,
	RELEASETIME,
	FMODFREQ,
	FMODINT,
	PMODFREQ,
	PMODINT
};

#endif