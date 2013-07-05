// onlinebin/java-online-audio-client/src/TextGrid.java

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


public class TextGrid extends Segmentation {

	public TextGrid()
	{
		
	}
	
	public TextGrid(Segmentation segmentation) {
		this.tiers = segmentation.tiers;
	}	

	@Override
	public void read(File file) throws IOException {

		throw new IOException("NYI");

	}

	@Override
	public void write(File file) throws IOException {

		PrintWriter writer = new PrintWriter(file);

		writer.println("File type = \"ooTextFile\"");
		writer.println("Object class = \"TextGrid\"");
		writer.println();
		writer.println("xmin = " + min());
		writer.println("xmax = " + max());
		writer.println("tiers? <exists>");
		writer.println("size = " + tiers.size());
		writer.println("item []:");

		for (int i = 0; i < tiers.size(); i++) {
			Tier tier = tiers.get(i);

			writer.println("\titem [" + (i + 1) + "]:");

			writer.println("\t\tclass = \"IntervalTier\"");
			writer.println("\t\tname = \"" + tier.name + "\"");

			writer.println("\t\txmin = " + tier.min());
			writer.println("\t\txmax = " + tier.max());
			writer.println("\t\tintervals: size = " + tier.segments.size());

			for (int j = 0; j < tier.segments.size(); j++) {
				Segment segment = tier.segments.get(j);

				writer.println("\t\tintervals [" + j + "]:");
				writer.println("\t\t\txmin = " + segment.start_time);
				writer.println("\t\t\txmax = " + segment.end_time);
				writer.println("\t\t\ttext = \"" + segment.name + "\"");
			}
		}

		writer.close();

	}
}
