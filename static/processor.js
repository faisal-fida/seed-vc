class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = (event) => {
      // Handle messages from the main thread if necessary
    };
  }

  process(inputs, outputs, parameters) {
    const inputChannelData = inputs[0][0]; // First input, first channel
    if (inputChannelData) {
      // Copy the input data
      const inputBuffer = inputChannelData.slice();
      // Send the audio data to the main thread
      this.port.postMessage(inputBuffer.buffer);
    }
    return true;
  }
}

registerProcessor('voice-processor', VoiceProcessor);