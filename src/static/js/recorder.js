// WAV audio encoder for browser microphone recording
// Encodes raw Web Audio PCM float32 buffers directly into standard 16-bit PCM WAV (16 kHz mono)
// Eliminates any need for FFmpeg or external browser transcoding libraries.

class WavRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.analyser = null;
        this.scriptProcessor = null;
        this.recordedChunks = [];
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
        this.targetSampleRate = 16000;
        this.onWaveformCallback = null;
        this.onTimerCallback = null;
    }

    async init() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Microphone access is not supported in this browser. Please use Chrome, Edge, Firefox, or Safari, or use the file upload fallback.");
        }
    }

    async start(onWaveform, onTimer) {
        this.onWaveformCallback = onWaveform;
        this.onTimerCallback = onTimer;
        this.recordedChunks = [];

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });
        } catch (err) {
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                throw new Error("Microphone permission was denied. Please allow microphone access in your browser bar, or use file upload.");
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                throw new Error("No microphone device was detected on your computer. Please connect a microphone or use file upload.");
            }
            throw new Error(`Microphone access error: ${err.message || err.name}`);
        }

        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioContextClass();
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);

        // Analyser for real-time visualizer
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        source.connect(this.analyser);

        // Buffer collection
        const bufferSize = 4096;
        this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
        this.scriptProcessor.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            this.recordedChunks.push(new Float32Array(inputData));
        };

        source.connect(this.scriptProcessor);
        this.scriptProcessor.connect(this.audioContext.destination);

        this.isRecording = true;
        this.startTime = Date.now();

        // Timer loop
        this.timerInterval = setInterval(() => {
            if (!this.isRecording) return;
            const elapsedMs = Date.now() - this.startTime;
            if (this.onTimerCallback) {
                this.onTimerCallback(elapsedMs / 1000);
            }
        }, 100);

        // Waveform loop
        this._drawLoop();
    }

    _drawLoop() {
        if (!this.isRecording) return;
        if (this.analyser && this.onWaveformCallback) {
            const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.analyser.getByteTimeDomainData(dataArray);
            this.onWaveformCallback(dataArray);
        }
        requestAnimationFrame(() => this._drawLoop());
    }

    async stop() {
        if (!this.isRecording) return null;
        this.isRecording = false;

        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }

        const inputSampleRate = this.audioContext ? this.audioContext.sampleRate : 44100;
        if (this.audioContext && this.audioContext.state !== 'closed') {
            await this.audioContext.close();
            this.audioContext = null;
        }

        // Merge float32 buffers
        let totalLength = 0;
        for (const chunk of this.recordedChunks) {
            totalLength += chunk.length;
        }

        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of this.recordedChunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
        }

        // Downsample to 16 kHz for ECAPA-TDNN standard pipeline
        const downsampled = this._resampleTo16kHz(merged, inputSampleRate, this.targetSampleRate);
        const durationSec = downsampled.length / this.targetSampleRate;

        // Encode as RIFF/WAV 16-bit PCM
        const wavBuffer = this._encodeWAV(downsampled, this.targetSampleRate);
        const blob = new Blob([wavBuffer], { type: 'audio/wav' });

        return {
            blob: blob,
            durationSec: durationSec,
            sampleRate: this.targetSampleRate
        };
    }

    _resampleTo16kHz(audioData, sourceSampleRate, targetSampleRate) {
        if (sourceSampleRate === targetSampleRate) return audioData;
        const ratio = sourceSampleRate / targetSampleRate;
        const newLength = Math.round(audioData.length / ratio);
        const result = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
            const originalIndex = i * ratio;
            const index = Math.floor(originalIndex);
            const fraction = originalIndex - index;
            const nextIndex = Math.min(index + 1, audioData.length - 1);
            result[i] = audioData[index] * (1 - fraction) + audioData[nextIndex] * fraction;
        }
        return result;
    }

    _encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        /* RIFF identifier */
        this._writeString(view, 0, 'RIFF');
        /* file length */
        view.setUint32(4, 36 + samples.length * 2, true);
        /* RIFF type */
        this._writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        this._writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (1 = PCM) */
        view.setUint16(20, 1, true);
        /* channel count (1 = mono) */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sampleRate, true);
        /* byte rate (sample rate * block align) */
        view.setUint32(28, sampleRate * 2, true);
        /* block align (channel count * bytes per sample) */
        view.setUint16(32, 2, true);
        /* bits per sample (16 bit) */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        this._writeString(view, 36, 'data');
        /* data chunk length */
        view.setUint32(40, samples.length * 2, true);

        // Convert float32 to 16-bit PCM
        let index = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            index += 2;
        }

        return buffer;
    }

    _writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
}

window.WavRecorder = WavRecorder;
