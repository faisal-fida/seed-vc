// File: /static/player-processor.js

class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.port.onmessage = (event) => {
      const newData = new Float32Array(event.data);
      const tempBuffer = new Float32Array(this.buffer.length + newData.length);
      tempBuffer.set(this.buffer);
      tempBuffer.set(newData, this.buffer.length);
      this.buffer = tempBuffer;
    };
  }

  process(inputs, outputs, parameters) {
    const outputChannelData = outputs[0][0]; // First output, first channel
    const outputChannelLength = outputChannelData.length;

    if (this.buffer.length >= outputChannelLength) {
      outputChannelData.set(this.buffer.subarray(0, outputChannelLength));
      this.buffer = this.buffer.subarray(outputChannelLength);
    } else {
      // Not enough data, fill with zeros to prevent clicks
      outputChannelData.fill(0);
    }
    return true;
  }
}

registerProcessor('player-processor', PlayerProcessor);