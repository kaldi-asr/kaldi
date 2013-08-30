// onlinebin/java-online-audio-client/src/Main.java

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

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.UIManager;

@SuppressWarnings(value={"serial","unchecked"})
public class Main extends JFrame {

	public static Main main_frame = null;

	private KaldiASR asr = null;
	private MLF mlf = null;

	private JComboBox tfInputFile;
	private JRadioButton rbFile, rbScp;
	private JCheckBox cbMLF, cbHTK, cbTextGrid, cbWebVTT;
	private JTextArea taLog;
	private JProgressBar progress;

	Main() {
		super("KALDI Online Audio Client");

		JPanel pMain = new JPanel();
		pMain.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
		add(pMain);

		pMain.setLayout(new BoxLayout(pMain, BoxLayout.PAGE_AXIS));

		JPanel pInput = new JPanel();
		pInput.setMaximumSize(new Dimension(Short.MAX_VALUE, 25));
		pInput.setLayout(new BoxLayout(pInput, BoxLayout.LINE_AXIS));
		pInput.add(new JLabel("Input:"));
		tfInputFile = new JComboBox();
		tfInputFile.setEditable(true);
		pInput.add(tfInputFile);
		rbFile = new JRadioButton("RAW");
		rbScp = new JRadioButton("SCP");
		ButtonGroup bgFile = new ButtonGroup();
		bgFile.add(rbFile);
		bgFile.add(rbScp);
		rbFile.setSelected(true);
		pInput.add(rbFile);
		pInput.add(rbScp);

		JPanel pOutput = new JPanel();
		pOutput.setMaximumSize(new Dimension(Short.MAX_VALUE, 25));
		pOutput.setLayout(new BoxLayout(pOutput, BoxLayout.LINE_AXIS));
		pOutput.add(new JLabel("Output:"));
		cbMLF = new JCheckBox("MLF");
		cbHTK = new JCheckBox("HTK");
		cbTextGrid = new JCheckBox("TextGrid");
		cbWebVTT = new JCheckBox("WebVTT");
		pOutput.add(cbMLF);
		pOutput.add(cbHTK);
		pOutput.add(cbTextGrid);
		pOutput.add(cbWebVTT);

		JPanel pRun = new JPanel();
		pRun.setMaximumSize(new Dimension(Short.MAX_VALUE, 25));
		pRun.setLayout(new BoxLayout(pRun, BoxLayout.LINE_AXIS));
		JButton bRun = new JButton("Run");
		bRun.setMaximumSize(new Dimension(Short.MAX_VALUE, 25));
		JButton bStop = new JButton("Stop");
		bStop.setMaximumSize(new Dimension(Short.MAX_VALUE, 25));
		pRun.add(bRun);
		pRun.add(bStop);

		taLog = new JTextArea();
		JScrollPane spLog = new JScrollPane(taLog);

		progress = new JProgressBar();

		pMain.add(pInput);
		pMain.add(pOutput);
		pMain.add(pRun);
		pMain.add(spLog);
		pMain.add(progress);

		File curr_dir = new File(".");
		String filenames[] = curr_dir.list();
		for (String name : filenames) {
			if (name.endsWith(".raw") || name.endsWith(".scp"))
				tfInputFile.addItem(name);
		}

		bRun.addActionListener(new AbstractAction() {
			@Override
			public void actionPerformed(ActionEvent arg0) {

				if (asr != null)
					return;

				new Thread() {
					public void run() {
						try {

							asr = new KaldiASR(Options.KALDI_HOST, Options.KALDI_PORT);
							if (cbMLF.isSelected())
								mlf = new MLF();

							if (rbFile.isSelected()) {

								recognize(new File(tfInputFile.getSelectedItem().toString()));

							} else if (rbScp.isSelected()) {

								SCPFile scp = new SCPFile(new File(tfInputFile.getSelectedItem().toString()));
								for (String filename : scp) {
									recognize(new File(filename));
								}

							}

							if (asr != null) {
								asr.close();
								asr = null;
							}

							if (cbMLF.isSelected())
								mlf.save(new File(Options.MLF_FILENAME));

						} catch (Exception e) {
							Main.error(e);
						}
					}
				}.start();

			}
		});

		bStop.addActionListener(new AbstractAction() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (asr != null) {
					try {
						asr.close();
					} catch (IOException e1) {
					}
					asr = null;
				}
			}
		});

		setDefaultCloseOperation(EXIT_ON_CLOSE);
		setBounds(250, 250, 600, 600);
		setVisible(true);
	}

	public static void log(String msg) {
		main_frame.taLog.append(msg + "\n");
	}

	public static void error(Throwable e) {
		main_frame.taLog.append("ERROR: " + e.getMessage() + "\n");
		e.printStackTrace();
	}

	public static void resetProgress(int max) {
		main_frame.progress.setMaximum(max);
		main_frame.progress.setValue(0);
	}

	public static void progress(int val) {
		main_frame.progress.setValue(val);
	}

	public void recognize(File raw_file) {

		try {
			String path = raw_file.getAbsolutePath();

			log("Recognizing: " + path);

			long input_size = raw_file.length();
			FileInputStream file_stream = new FileInputStream(raw_file);

			OutputProcess output = new OutputProcess();

			if (cbTextGrid.isSelected()) {
				File textgrid = new File(path.substring(0, path.length() - 4) + ".TextGrid");
				output.saveTextGrid(textgrid);
			}

			if (cbWebVTT.isSelected()) {
				File webvtt = new File(path.substring(0, path.length() - 4) + ".vtt");
				output.saveWebVVT(webvtt);
			}

			if (cbHTK.isSelected()) {
				File htk = new File(path.substring(0, path.length() - 4) + ".lab");
				output.saveHTK(htk);
			}

			asr.recognize(file_stream, input_size, output);

			output.finalize();

			if (cbMLF.isSelected()) {
				String filename = raw_file.getName();
				mlf.add(filename.substring(0, filename.length() - 4) + ".lab", output);
			}

			file_stream.close();

		} catch (Exception e) {
			Main.error(e);
		}
	}

	public static void main(String[] args) {

		try {

			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

			if (Options.propertiesFileExists())
				Options.load();
			else
				Options.save();

			main_frame = new Main();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
