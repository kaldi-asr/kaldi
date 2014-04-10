// onlinebin/java-online-audio-client/src/Options.java

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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public class Options {

	public static File PROPERTY_FILE = new File("client.properties");

	public static String KALDI_HOST = "10.4.4.23";

	public static int KALDI_PORT = 5010;

	public static String KALDI_ENCODING = "CP1250";

	public static int AUDIO_PACKET_SIZE = 1024;

	public static String MLF_FILENAME = "out.mlf";

	public static int SUBTITLES_CHARS_PER_CUE = 64;

	public static float SUBTITLES_TIME_DIFF = 2.0f;

	public static float SUBTITLES_KEEP_ALIVE = 1.0f;

	public static boolean SUBTITLES_WORD_BY_WORD = true;

	public static void save() throws FileNotFoundException, IOException {
		Properties properties = new Properties();

		properties.put("KALDI_HOST", KALDI_HOST);
		properties.put("KALDI_PORT", "" + KALDI_PORT);
		properties.put("KALDI_ENCODING", KALDI_ENCODING);
		properties.put("AUDIO_PACKET_SIZE", "" + AUDIO_PACKET_SIZE);
		properties.put("MLF_FILENAME", MLF_FILENAME);
		properties.put("SUBTITLES_CHARS_PER_CUE", "" + SUBTITLES_CHARS_PER_CUE);
		properties.put("SUBTITLES_TIME_DIFF", "" + SUBTITLES_TIME_DIFF);
		properties.put("SUBTITLES_KEEP_ALIVE", "" + SUBTITLES_KEEP_ALIVE);
		if (SUBTITLES_WORD_BY_WORD)
			properties.put("SUBTITLES_WORD_BY_WORD", "TRUE");
		else
			properties.put("SUBTITLES_WORD_BY_WORD", "FALSE");

		properties.storeToXML(new FileOutputStream(PROPERTY_FILE), "Automatically generated property file.");
	}

	public static void load() throws FileNotFoundException, IOException, NumberFormatException {
		Properties properties = new Properties();

		properties.loadFromXML(new FileInputStream(PROPERTY_FILE));

		if (properties.containsKey("KALDI_HOST"))
			KALDI_HOST = properties.getProperty("KALDI_HOST");
		if (properties.containsKey("KALDI_PORT"))
			KALDI_PORT = Integer.parseInt(properties.getProperty("KALDI_PORT"));
		if (properties.containsKey("KALDI_ENCODING"))
			KALDI_ENCODING = properties.getProperty("KALDI_ENCODING");
		if (properties.containsKey("AUDIO_PACKET_SIZE"))
			AUDIO_PACKET_SIZE = Integer.parseInt(properties.getProperty("AUDIO_PACKET_SIZE"));
		if (properties.containsKey("MLF_FILENAME"))
			MLF_FILENAME = properties.getProperty("MLF_FILENAME");
		if (properties.containsKey("SUBTITLES_CHARS_PER_CUE"))
			SUBTITLES_CHARS_PER_CUE = Integer.parseInt(properties.getProperty("SUBTITLES_CHARS_PER_CUE"));
		if (properties.containsKey("SUBTITLES_TIME_DIFF"))
			SUBTITLES_TIME_DIFF = Float.parseFloat(properties.getProperty("SUBTITLES_TIME_DIFF"));
		if (properties.containsKey("SUBTITLES_KEEP_ALIVE"))
			SUBTITLES_KEEP_ALIVE = Float.parseFloat(properties.getProperty("SUBTITLES_KEEP_ALIVE"));
		if (properties.containsKey("SUBTITLES_WORD_BY_WORD")) {
			if (properties.getProperty("SUBTITLES_WORD_BY_WORD").equals("TRUE"))
				SUBTITLES_WORD_BY_WORD = true;
			else
				SUBTITLES_WORD_BY_WORD = false;
		}

	}

	public static boolean propertiesFileExists() {
		return PROPERTY_FILE.exists() && PROPERTY_FILE.canRead();
	}
}
