import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Vector;

public class WebVTT {

	public static class Cue {
		float start = -Float.MAX_VALUE, end = -Float.MAX_VALUE;// in seconds
		String text = "";

		Cue() {
		}

		Cue(Cue copy) {
			this.start = copy.start;
			this.end = copy.end;
			this.text = copy.text;
		}

	}

	private Vector<Cue> cues = new Vector<Cue>();

	Cue last_cue = new Cue();
	float last_cue_end = -Float.MAX_VALUE;

	public void addWord(String word, float start, float end) {

		if (Options.SUBTITLES_WORD_BY_WORD) {

			if ((last_cue.text.length() + word.length() + 1) > Options.SUBTITLES_CHARS_PER_CUE
					|| (start - last_cue.end) > Options.SUBTITLES_TIME_DIFF) {
				{
					last_cue_end = last_cue.end;
					last_cue = new Cue();
				}
			}

			if (last_cue.end >= 0)
				last_cue.start = last_cue.end;
			else
				last_cue.start = start;

			if (last_cue.start <= last_cue_end)// this is the old cue end!
				last_cue.start = last_cue_end + 0.1f;

			last_cue.end = end;
			if (last_cue.text.length() > 0)
				last_cue.text += " ";
			last_cue.text += word;

			cues.add(new Cue(last_cue));

		} else {
			if ((last_cue.text.length() + word.length() + 1) > Options.SUBTITLES_CHARS_PER_CUE
					|| (start - last_cue.end) > Options.SUBTITLES_TIME_DIFF) {
				if (last_cue.text.length() > 0) {
					last_cue_end = last_cue.end;
					cues.add(last_cue);
					last_cue = new Cue();
				}
			}

			if (last_cue.text.length() == 0)
				last_cue.start = start;
			else
				last_cue.text += " ";

			if (last_cue.start <= last_cue_end)// this is the old cue end!
				last_cue.start = last_cue_end + 0.1f;

			last_cue.text += word;
			last_cue.end = end;
		}
	}

	public void finalize() {

		if (!Options.SUBTITLES_WORD_BY_WORD) {
			if (last_cue.text.length() > 0) {
				cues.add(last_cue);
				last_cue = new Cue();
			}
		}

		float diff;
		for (int i = 0; i < cues.size() - 1; i++) {
			diff = cues.get(i + 1).start - cues.get(i).end - 0.1f;
			if (diff > Options.SUBTITLES_KEEP_ALIVE)
				diff = Options.SUBTITLES_KEEP_ALIVE;
			cues.get(i).end += diff;
		}

		cues.get(cues.size() - 1).end += Options.SUBTITLES_KEEP_ALIVE;

	}

	public void save(File file) throws FileNotFoundException {
		finalize();

		PrintWriter writer = new PrintWriter(file);

		writer.println("WEBVTT FILE");
		writer.println();

		int i = 1;
		for (Cue cue : cues) {
			writer.println("" + i);
			if (Options.SUBTITLES_WORD_BY_WORD)
				writer.println(timeToTimecode(cue.start) + " --> " + timeToTimecode(cue.end)
						+ " align:start position:0%");
			else
				writer.println(timeToTimecode(cue.start) + " --> " + timeToTimecode(cue.end));
			writer.println(cue.text);
			writer.println();
			i++;
		}

		writer.close();
	}

	private String timeToTimecode(float time) {
		int h, m, s, ms;
		s = (int) time;
		ms = (int) ((time - (float) s) * 1000.0f);
		m = s / 60;
		s %= 60;
		h = m / 60;
		m %= 60;

		return String.format("%02d:%02d:%02d.%03d", h, m, s, ms);
	}
}
