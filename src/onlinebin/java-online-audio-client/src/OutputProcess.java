// onlinebin/java-online-audio-client/src/OutputProcess.java

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

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Vector;

public class OutputProcess {

	public static class Word {
		public String word;
		public float start;
		public float end;

		public Word(String word, float start, float end) {
			this.word = word;
			this.start = start;
			this.end = end;
		}
	}

	public Vector<Word> words = new Vector<Word>();
	private File textgrid_file = null;
	private File webvvt_file = null;
	private File htk_file = null;

	public void addWord(Word word) {
		words.add(word);
	}

	public void saveTextGrid(File file) {
		textgrid_file = file;
	}

	public void saveHTK(File file) {
		htk_file = file;
	}

	public void saveWebVVT(File file) {
		webvvt_file = file;
	}

	public void finalize() throws IOException {
		if (textgrid_file != null) {
			TextGrid textgrid = new TextGrid();

			for (Word word : words) {
				textgrid.addSegment(0, word.start, word.end, word.word);
			}

			textgrid.tiers.get(0).name = "KALDI";

			textgrid.write(textgrid_file);
		}

		if (webvvt_file != null) {
			WebVTT webvvt = new WebVTT();

			for (Word word : words) {
				webvvt.addWord(word.word, word.start, word.end);
			}

			webvvt.save(webvvt_file);
		}

		if (htk_file != null) {
			PrintWriter htk = new PrintWriter(htk_file, Options.KALDI_ENCODING);
			for (Word word : words) {
				int s = (int) (word.start * 10000000);
				int e = (int) (word.end * 10000000);

				htk.println(s + " " + e + " " + word.word);
			}
			htk.close();
		}
	}

}
