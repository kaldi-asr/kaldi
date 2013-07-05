// onlinebin/java-online-audio-client/src/Segmentation.java

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
import java.util.LinkedList;
import java.util.List;

public abstract class Segmentation {

	public static class Segment {
		public double start_time;
		public double end_time;
		public String name;
	}

	public static class Tier {
		public String name;
		public List<Segment> segments = new LinkedList<Segment>();

		public double min() {
			double min = segments.get(0).start_time;
			for (Segment s : segments)
				if (min > s.start_time)
					min = s.start_time;
			return min;
		}

		public double max() {
			double max = segments.get(0).end_time;
			for (Segment s : segments)
				if (max < s.end_time)
					max = s.end_time;
			return max;
		}

		public void add(double start, double end, String name) {
			Segment segment = new Segment();
			segment.start_time = start;
			segment.end_time = end;
			segment.name = name;
			segments.add(segment);
		}
	}

	public List<Tier> tiers = new LinkedList<Tier>();

	public double min() {
		double min = tiers.get(0).min();
		for (Tier t : tiers) {
			double d = t.min();
			if (min > d)
				min = d;
		}
		return min;
	}

	public double max() {
		double max = tiers.get(0).max();
		for (Tier t : tiers) {
			double d = t.max();
			if (max < d)
				max = d;
		}
		return max;
	}

	public void addSegment(int tier, double start, double end, String name) {
		while (tiers.size() <= tier)
			tiers.add(new Tier());

		tiers.get(tier).add(start, end, name);
	}
	
	public void addTiers(Segmentation segmentation)
	{
		tiers.addAll(segmentation.tiers);
	}
	
	public abstract void read(File file) throws IOException;
	public abstract void write(File file) throws IOException;

}
