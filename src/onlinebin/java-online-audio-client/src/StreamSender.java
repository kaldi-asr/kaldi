// onlinebin/java-online-audio-client/src/StreamSender.java

// Copyright 2013 Polish-Japanese Institute of Information Technology (author: Danijel Korzinek)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class StreamSender implements Runnable {

	private InputStream input_stream;
	private long input_size;
	private OutputStream output_stream;

	public StreamSender(InputStream input_stream, long input_size, OutputStream output_stream) {
		this.input_stream = input_stream;
		this.input_size = input_size;
		this.output_stream = output_stream;
	}

	@Override
	public void run() {

		try {

			byte buffer[] = new byte[Options.AUDIO_PACKET_SIZE];

			ByteBuffer size_buf = ByteBuffer.allocate(4);
			size_buf.order(ByteOrder.LITTLE_ENDIAN);

			long input_processed = 0;

			Main.resetProgress((int) input_size);

			while (true) {
				int read = input_stream.read(buffer);

				if (read < 0)
					break;

				if ((read & 1) != 0) {
					read--;
				}

				size_buf.putInt(0, read);

				output_stream.write(size_buf.array(), 0, 4);

				output_stream.write(buffer, 0, read);

				output_stream.flush();

				input_processed += read;

				Main.progress((int) input_processed);
			}

			size_buf.putInt(0, 0);
			output_stream.write(size_buf.array(), 0, 4);

		} catch (Exception e) {
			Main.error(e);
		}

	}

}
