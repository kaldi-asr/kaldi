// onlinebin/java-online-audio-client/src/MLF.java

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
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Vector;

public class MLF {

	public static class LabFile {
		String filename;
		Vector<String> words = new Vector<String>();
	}

	private Vector<LabFile> labels = new Vector<LabFile>();

	public void add(String filename, OutputProcess outputs) {
		LabFile file = new LabFile();

		file.filename = filename;
		for (OutputProcess.Word word : outputs.words) {
			file.words.add(word.word);
		}

		labels.add(file);
	}

	public void save(File file) throws FileNotFoundException, UnsupportedEncodingException {

		PrintWriter writer = new PrintWriter(file, Options.KALDI_ENCODING);

		writer.println("#!MLF!#");

		for (LabFile labfile : labels) {
			writer.println("\"*/" + labfile.filename + "\"");

			for (String word : labfile.words)
				writer.println(word);

			writer.println(".");
		}

		writer.close();
	}
}
