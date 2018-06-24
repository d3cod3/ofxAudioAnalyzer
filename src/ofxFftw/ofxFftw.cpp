#include "ofxFftw.h"


ofxFftw::ofxFftw(){

}

ofxFftw::~ofxFftw(){
	delete [] signal;
	delete [] real;
	delete [] imag;
	delete [] amplitude;
	delete [] phase;
	delete [] window;
	delete [] inverseWindow;
	
	if (fftPlan != NULL) {
		fftwf_destroy_plan(fftPlan);
		fftwf_free(fftIn);
		fftwf_free(fftOut);

		fftwf_destroy_plan(ifftPlan);
		fftwf_free(ifftIn);
		fftwf_free(ifftOut);
		fftwf_cleanup();
	}
}

void ofxFftw::setup(int signalSize, fftWindowType windowType) {
	this->signalSize = signalSize;
	this->binSize = (signalSize / 2) + 1;

	signalNormalized = true;
	signal = new float[signalSize];

	cartesianUpdated = true;
	cartesianNormalized = true;
	real = new float[binSize];
	imag = new float[binSize];

	polarUpdated = true;
	polarNormalized = true;
	amplitude = new float[binSize];
	phase = new float[binSize];

	clear();

	window = new float[signalSize];
	inverseWindow = new float[signalSize];
	setWindowType(windowType);

	fftIn = (float*) fftwf_malloc(sizeof(float) * signalSize);
	fftOut = (float*) fftwf_malloc(sizeof(float) * signalSize);
	fftPlan = fftwf_plan_r2r_1d(signalSize, fftIn, fftOut, FFTW_R2HC, FFTW_DESTROY_INPUT | FFTW_MEASURE);

	ifftIn = (float*) fftwf_malloc(sizeof(float) * signalSize);
	ifftOut = (float*) fftwf_malloc(sizeof(float) * signalSize);
	ifftPlan = fftwf_plan_r2r_1d(signalSize, ifftIn, ifftOut, FFTW_HC2R, FFTW_DESTROY_INPUT | FFTW_MEASURE);
}

int ofxFftw::getBinSize() {
	return binSize;
}

int ofxFftw::getSignalSize() {
	return signalSize;
}

void ofxFftw::setWindowType(fftWindowType windowType) {
	this->windowType = windowType;
	if(windowType == OF_FFT_WINDOW_RECTANGULAR) {
		for(int i = 0; i < signalSize; i++)
			window[i] = 1; // only used for windowSum
	} else if(windowType == OF_FFT_WINDOW_BARTLETT) {
		int half = signalSize / 2;
		for (int i = 0; i < half; i++) {
			window[i] = ((float) i / half);
			window[i + half] = (1 - ((float) i / half));
		}
	} else if(windowType == OF_FFT_WINDOW_HANN) {
		for(int i = 0; i < signalSize; i++)
			window[i] = .5 * (1 - cos((TWO_PI * i) / (signalSize - 1)));
	} else if(windowType == OF_FFT_WINDOW_HAMMING) {
		for(int i = 0; i < signalSize; i++)
			window[i] = .54 - .46 * cos((TWO_PI * i) / (signalSize - 1));
	} else if(windowType == OF_FFT_WINDOW_SINE) {
		for(int i = 0; i < signalSize; i++)
			window[i] = sin((PI * i) / (signalSize - 1));
	}

	windowSum = 0;
	for(int i = 0; i < signalSize; i++)
		windowSum += window[i];

	for(int i = 0; i < signalSize; i++)
		inverseWindow[i] = 1. / window[i];
}

void ofxFftw::clear() {
	memset(signal, 0, sizeof(float) * signalSize);
	memset(real, 0, sizeof(float) * binSize);
	memset(imag, 0, sizeof(float) * binSize);
	memset(amplitude, 0, sizeof(float) * binSize);
	memset(phase, 0, sizeof(float) * binSize);
}

void ofxFftw::copySignal(const float* signal) {
	memcpy(this->signal, signal, sizeof(float) * signalSize);
}

void ofxFftw::copyReal(float* real) {
	memcpy(this->real, real, sizeof(float) * binSize);
}

void ofxFftw::copyImaginary(float* imag) {
	if(imag == NULL)
		memset(this->imag, 0, sizeof(float) * binSize);
	else
		memcpy(this->imag, imag, sizeof(float) * binSize);
}

void ofxFftw::copyAmplitude(float* amplitude) {
	memcpy(this->amplitude, amplitude, sizeof(float) * binSize);
}

void ofxFftw::copyPhase(float* phase) {
	if(phase == NULL)
		memset(this->phase, 0, sizeof(float) * binSize);
	else
		memcpy(this->phase, phase, sizeof(float) * binSize);
}

void ofxFftw::prepareSignal() {
	if(!signalUpdated)
		updateSignal();
	if(!signalNormalized)
		normalizeSignal();
}

void ofxFftw::updateSignal() {
	prepareCartesian();
	executeIfft();
	signalUpdated = true;
	signalNormalized = false;
}

void ofxFftw::normalizeSignal() {
	float normalizer = (float) windowSum / (2 * signalSize);
	for (int i = 0; i < signalSize; i++)
		signal[i] *= normalizer;
	signalNormalized = true;
}

float* ofxFftw::getSignal() {
	prepareSignal();
	return signal;
}

void ofxFftw::clampSignal() {
	prepareSignal();
	for(int i = 0; i < signalSize; i++) {
		if(signal[i] > 1)
			signal[i] = 1;
		else if(signal[i] < -1)
			signal[i] = -1;
	}
}

void ofxFftw::prepareCartesian() {
	if(!cartesianUpdated) {
		if(!polarUpdated)
			executeFft();
		else
			updateCartesian();
	}
	if(!cartesianNormalized)
		normalizeCartesian();
}

float* ofxFftw::getReal() {
	prepareCartesian();
	return real;
}

float* ofxFftw::getImaginary() {
	prepareCartesian();
	return imag;
}

void ofxFftw::preparePolar() {
	if(!polarUpdated)
		updatePolar();
	if(!polarNormalized)
		normalizePolar();
}

float* ofxFftw::getAmplitude() {
	preparePolar();
	return amplitude;
}

float* ofxFftw::getPhase() {
	preparePolar();
	return phase;
}

float ofxFftw::getAmplitudeAtBin(float bin) {
	float* amplitude = getAmplitude();
	int lowBin = ofClamp(floorf(bin), 0, binSize - 1);
	int highBin = ofClamp(ceilf(bin), 0, binSize - 1);
	return ofMap(bin, lowBin, highBin, amplitude[lowBin], amplitude[highBin]);
}

float ofxFftw::getBinFromFrequency(float frequency, float sampleRate) {
	return frequency * binSize / (sampleRate / 2);
}

float ofxFftw::getAmplitudeAtFrequency(float frequency, float sampleRate) {
	return getAmplitudeAtBin(getBinFromFrequency(frequency, sampleRate));
}

void ofxFftw::updateCartesian() {
	for(int i = 0; i < binSize; i++) {
		real[i] = cosf(phase[i]) * amplitude[i];
		imag[i] = sinf(phase[i]) * amplitude[i];
	}
	cartesianUpdated = true;
	cartesianNormalized = polarNormalized;
}

void ofxFftw::normalizeCartesian() {
	float normalizer = 2. / windowSum;
	for(int i = 0; i < binSize; i++) {
		real[i] *= normalizer;
		imag[i] *= normalizer;
	}
	cartesianNormalized = true;
}

void ofxFftw::updatePolar() {
	prepareCartesian();
	for(int i = 0; i < binSize; i++) {
		amplitude[i] = cartesianToAmplitude(real[i], imag[i]);
		phase[i] = cartesianToPhase(real[i], imag[i]);
	}
	polarUpdated = true;
	polarNormalized = cartesianNormalized;
}

void ofxFftw::normalizePolar() {
	float normalizer = 2. / windowSum;
	for(int i = 0; i < binSize; i++)
		amplitude[i] *= normalizer;
	polarNormalized = true;
}

void ofxFftw::clearUpdates() {
	cartesianUpdated = false;
	polarUpdated = false;
	cartesianNormalized = false;
	polarNormalized = false;
	signalUpdated = false;
	signalNormalized = false;
}

void ofxFftw::setSignal(const vector<float>& signal) {
	setSignal(&signal[0]);
}

void ofxFftw::setSignal(const float* signal) {
	clearUpdates();
	copySignal(signal);
	signalUpdated = true;
	signalNormalized = true;
}

void ofxFftw::setCartesian(float* real, float* imag) {
	clearUpdates();
	copyReal(real);
	copyImaginary(imag);
	cartesianUpdated = true;
	cartesianNormalized = true;
}

void ofxFftw::setPolar(float* amplitude, float* phase) {
	clearUpdates();
	copyAmplitude(amplitude);
	copyPhase(phase);
	polarUpdated = true;
	polarNormalized = true;
}

void ofxFftw::executeFft() {
	memcpy(fftIn, signal, sizeof(float) * signalSize);
	runWindow(fftIn);
	fftwf_execute(fftPlan);
	// explanation of halfcomplex format:
	// http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
	copyReal(fftOut);
	imag[0] = 0;
	for (int i = 1; i < binSize; i++)
		imag[i] = fftOut[signalSize - i];
	cartesianUpdated = true;
}

void ofxFftw::executeIfft() {
	memcpy(ifftIn, real, sizeof(float) * binSize);
	for (int i = 1; i < binSize; i++)
		ifftIn[signalSize - i] = imag[i];
	fftwf_execute(ifftPlan);
	runInverseWindow(ifftOut);
	copySignal(ifftOut);
}
