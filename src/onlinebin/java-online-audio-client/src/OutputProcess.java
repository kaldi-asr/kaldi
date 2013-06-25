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
				int s = (int) (word.start * Options.AUDIO_SAMPLING_FREQUENCY);
				int e = (int) (word.end * Options.AUDIO_SAMPLING_FREQUENCY);

				htk.println(s + " " + e + " " + word.word + " 0.0");
			}
			htk.close();
		}
	}

}
