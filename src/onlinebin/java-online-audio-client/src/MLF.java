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
