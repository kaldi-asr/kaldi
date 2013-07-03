// onlinebin/java-online-audio-client/src/SCPFile.java

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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Vector;

public class SCPFile implements Iterable<String> {

	Vector<String> items;

	SCPFile(File file) throws IOException {
		items = new Vector<String>();

		BufferedReader reader = new BufferedReader(new FileReader(file));

		String line;

		while ((line = reader.readLine()) != null) {
			line = line.trim();
			if (line.length() > 0)
				items.add(line);
		}

		reader.close();
	}

	@Override
	public Iterator<String> iterator() {
		return items.iterator();
	}

}
