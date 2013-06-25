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
