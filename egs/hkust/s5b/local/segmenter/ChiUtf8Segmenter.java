//
// Copyright 2013-2014, Hong Kong University of Science and Technology (author: Ricky Chan Ho Yin)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// This is a Chinese word segmentation program, it allows operation with 5 modes. 
// It assumes input Chinese characters with UTF-8 encoding.
//
// Usage: java ChiUtf8Segmenter [-mode1|-mode2|-mode3|-mode4|-mode5] input_file wordprob_map [numbers_identity_file]
//
// Default option:         left longest segments
// Option: -mode1          left longest segments
// Option: -mode2          right longest segments
// Option: -mode3          choose segments from left longest or right longest (which give less segments)
// Option: -mode4          segments with higher unigram probability of left longest or right longest
// Option: -mode5          Viterbi search segmentation (by unigram log probability path cost)
//
// input_file              name of input file for segmentation
// wordprob_map            wordlist with log probabilities for segmentation
// numbers_identity_file   name of file for numbers identification (optional input)


import java.io.*;
import java.lang.*;
import java.util.*;
import java.util.regex.*;

public class ChiUtf8Segmenter {
  private final String encoding = "UTF-8";
  private final int maxword_length = 8;
  private final String seg_separator = " ";
  private final float unkstr_prob = -12.0f;

  private HashMap<String, Float> wordprob_mapdata = null;  // "name to negative cost" or "name to log probability" table
  private WordProbMap wordprobmap = null;
  private TreeSet<String> numbers_data = null;

  private static int running_mode;
    
  private static ArrayList<String> segstrLeftBuffer = new ArrayList<String>();
  private static LinkedList<String> segstrRightBuffer = new LinkedList<String>();

  public ChiUtf8Segmenter(String wordprobFile) throws IOException {
    wordprobmap = new WordProbMap(wordprobFile, encoding);
    wordprob_mapdata = wordprobmap.getProbMap();
  }

  public ChiUtf8Segmenter(String wordprobFile, String numbersFile) throws IOException {
    wordprobmap = new WordProbMap(wordprobFile, encoding);
    wordprob_mapdata = wordprobmap.getProbMap();

    numbers_data = new TreeSet<String>();
    loadChiRes(numbers_data, numbersFile);
  }

  public void cleanRes() {
    if(wordprobmap!=null) {
      wordprobmap.clearMap();
      wordprobmap = null;
    }

    if(numbers_data!=null) {
      numbers_data.clear();
      numbers_data = null;
    }
  }

  private void loadChiRes(TreeSet<String> resdata, String sourcefile) {
    String dataline;
    try {
      InputStream in = getClass().getResourceAsStream(sourcefile);
      BufferedReader rd = new BufferedReader(new InputStreamReader(in, encoding));

      dataline = rd.readLine();
      while(dataline != null) {
        dataline = dataline.trim();
        if(dataline.length() == 0) continue;
        resdata.add(dataline);
        dataline = rd.readLine();
      }
      in.close();
      rd.close();
    }
    catch (Exception e) {
      System.err.println("Load resources for "+sourcefile+" error: " + e);
    }
  }

  private boolean isNumber(String word) {
    String tmp;

    if(numbers_data == null) return false;

    int ll = word.length();
    if(ll == 0) return false;

    for(int i = 0; i<ll; i++) {
      tmp = word.substring(i, i+1);
      if(numbers_data.contains(tmp) == false) {
	return false;
      }
    }
    return true;
  }

  public String segmentLine(String cline, String separator, int mode) {
    int[] boundaries = null;
    int[] rboundaries = null;
    int i, lsepn, rsepn, clen;
    String concatStr = null;

    clen = cline.length();
    if(clen==0) return "";

    if(mode == 1) {
      boundaries = new int[clen];
      segmentLineLeftOffsets(cline, boundaries);
      if(boundaries.length == 0) { return cline; }
    }
    else if(mode == 2) {
      rboundaries = new int[clen];
      segmentLineRightOffsets(cline, rboundaries);
      if(rboundaries.length == 0) { return cline; }
    }
    else if(mode == 3 || mode == 4) {
      boundaries = new int[clen];
      rboundaries = new int[clen];
      segmentLineLeftOffsets(cline, boundaries);
      segmentLineRightOffsets(cline, rboundaries);
      if(boundaries.length == 0 && rboundaries.length == 0) { return cline; }
    }
    else {
    }

    if(mode == 1) {
      concatStr = concatLineLeft(cline, boundaries, separator);
    }
    else if(mode == 2) {
      concatStr = concatLineRight(cline, rboundaries, separator);
    }
    else if(mode == 3) {
      lsepn = rsepn = 0;
      for(i=0; i<boundaries.length; i++) { 
        if(boundaries[i] > 0) lsepn++;
      }
      for(i = rboundaries.length-1; i >= 0; i--) {
        if(rboundaries[i] > 0) rsepn++;
      }
      if(rsepn < lsepn) { // choose right
        concatStr = concatLineRight(cline, rboundaries, separator);
      }
      else {
        concatStr = concatLineLeft(cline, boundaries, separator);
      }
    }
    else if(mode == 4) {
      String tmpstr;
      float lvalue,rvalue;
      lvalue = rvalue = 0.0f;
      concatStr = "";
      
      concatLineLeft(cline, boundaries);
      concatLineRight(cline, rboundaries);

      for(i=0; i<segstrLeftBuffer.size(); i++) {
        tmpstr = segstrLeftBuffer.get(i);
        if(wordprob_mapdata.containsKey(tmpstr))
          lvalue += wordprob_mapdata.get(tmpstr);
        else lvalue += unkstr_prob;
      }

      ListIterator<String> listIterator = segstrRightBuffer.listIterator();
      while (listIterator.hasNext()) {
        tmpstr = listIterator.next();
        if(wordprob_mapdata.containsKey(tmpstr))
          rvalue += wordprob_mapdata.get(tmpstr);
        else rvalue += unkstr_prob;
      }

      if(lvalue >= rvalue) {
        for(i=0; i<segstrLeftBuffer.size(); i++) {
          concatStr += segstrLeftBuffer.get(i);
          concatStr += separator;
        }
      }
      else {
        listIterator = segstrRightBuffer.listIterator();
        while(listIterator.hasNext()) {
          concatStr += listIterator.next();
          concatStr += separator;
        }
      }
    }
    else if(mode == 5) {
      concatStr = viterbiSeg(cline, separator);
    }
    else {
      concatStr = ""; // to be implemented for other algorithm
    }

    return concatStr;
  }

  private String viterbiSeg(String cline, String separator) {
    int i, j=0;
    String segstr, substr = "";
    ArrayList<String> history_path = null;
    int history_num_element;
    float oldpath_prob, newpath_prob;
    SearchHistoryPath shp;
    boolean skip_flag;

    int clength = cline.length();
    if(clength < 1) return substr;

    ArrayList<SearchHistoryPath> bestState = new ArrayList<SearchHistoryPath>(clength);
    
    for(i=0; i<clength; i++) {
      bestState.add(new SearchHistoryPath());
    }

    i = -1;
    history_num_element = 0;
    oldpath_prob = 0.0f;

    while(i<clength-1) {
      if(i>-1 && bestState.get(i).getNumElement() == 0) {
        i++;
        continue;
      }
      if(i>-1) {
        history_num_element = bestState.get(i).getNumElement();
        history_path = bestState.get(i).getList();
        oldpath_prob = bestState.get(i).getLogProb(); 
      }

      skip_flag = false;
      if( (i+3 <= clength-1) && cline.substring(i+1, i+4).compareTo("<s>")==0 ) {
        j=3;
        substr = cline.substring(i+1, i+j+1);
        skip_flag = true;
      }
      else if( (i+4 <= clength-1) && cline.substring(i+1, i+5).compareTo("</s>")==0 ) {
        j=4;
        substr = cline.substring(i+1, i+j+1);
        skip_flag = true;
      }
      else if(Character.UnicodeBlock.of(cline.charAt(i+1)) == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) {
        j=1;
        substr = cline.substring(i+1, i+j+1);
        if(wordprob_mapdata.containsKey(substr))
          newpath_prob = oldpath_prob + wordprob_mapdata.get(substr);
        else 
          newpath_prob = oldpath_prob + unkstr_prob;
        if(bestState.get(i+j).getNumElement()==0 || bestState.get(i+j).getLogProb()<newpath_prob) {
          shp = new SearchHistoryPath();
          if(history_path != null) shp.setList(history_path);
          shp.addElement(substr, newpath_prob);
          bestState.set(i+j, shp);
        }

        for(j=2; j<=maxword_length && (i+j<clength); j++) {
          substr = cline.substring(i+1, i+j+1);
          if(wordprob_mapdata.containsKey(substr)) {
            newpath_prob = wordprob_mapdata.get(substr) + oldpath_prob;
            if(bestState.get(i+j).getNumElement()==0 || bestState.get(i+j).getLogProb()<newpath_prob) {
              shp = new SearchHistoryPath();
              if(history_path != null) shp.setList(history_path);
              shp.addElement(substr, newpath_prob);
              bestState.set(i+j, shp);
            }
          }
        }

      }
      else if(Character.isWhitespace(cline.charAt(i+1))) {
        j=1;
        while ( i+j < clength-1 && Character.isWhitespace(cline.charAt(i+j+1)) && (Character.UnicodeBlock.of(cline.charAt(i+j+1)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ) {
          j++;
        }
        substr = "";
        skip_flag = true;
      }
      else if(Character.isLetter(cline.charAt(i+1))) {
        j=1;
        while ( i+j < clength-1 && Character.isLetter(cline.charAt(i+j+1)) && (Character.UnicodeBlock.of(cline.charAt(i+j+1)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ) {
          j++;
        }
        substr = cline.substring(i+1, i+j+1);
        skip_flag = true;
      }
      else if(Character.isDigit(cline.charAt(i+1))) {
        j=1;
        while ( i+j < clength-1 && (Character.isDigit(cline.charAt(i+j+1)) || cline.charAt(i+j+1)=='.') && (Character.UnicodeBlock.of(cline.charAt(i+j+1)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ) {
          j++;
        }
        substr = cline.substring(i+1, i+j+1);
        skip_flag = true;
      }
      else {
        j=1;
        newpath_prob = oldpath_prob + unkstr_prob;
        substr = cline.substring(i+1, i+j+1);
        if(bestState.get(i+j).getNumElement()==0 || bestState.get(i+j).getLogProb()<newpath_prob) {
          shp = new SearchHistoryPath();
          if(history_path != null) shp.setList(history_path);
          shp.addElement(substr, newpath_prob);
          bestState.set(i+j, shp);
        }
      }
      
      if(skip_flag==true) {
        shp = new SearchHistoryPath();
        if(history_path != null) shp.setList(history_path);
        shp.addElement(substr, oldpath_prob);
        bestState.set(i+j, shp);
        i+=j;
      }
      else { i++; }
    }

    boolean former_num_flag = false;
    shp = bestState.get(i);
    segstr = "";
    ListIterator<String> listIterator = shp.getList().listIterator();
    while (listIterator.hasNext()) {
      substr = listIterator.next();
      if(substr.length()>0) {
        if(isNumber(substr)==false) {
          if(former_num_flag==true) segstr += separator;
          segstr += substr;
          segstr += separator;
          former_num_flag = false;
        }
        else {
          segstr += substr;
          former_num_flag = true;
        }
      }
    }

    shp = null;
    bestState = null;

    return segstr;
  }
  
  private static ArrayList<String> concatLineLeft(String cline, int [] boundaries) {
    int i;

    segstrLeftBuffer.clear();

    for(i=0; i<boundaries.length; i++) {
      if(boundaries[i] > 0) segstrLeftBuffer.add(cline.substring(i, i+boundaries[i]));
    }
   
    return segstrLeftBuffer;
  }

  private static LinkedList<String> concatLineRight(String cline, int [] boundaries) {
    int i;
    String substr;

    segstrRightBuffer.clear();

    for(i = boundaries.length-1; i >= 0; i--) {
      if(boundaries[i] > 0 && i-boundaries[i]+1 >= 0)
      {
        substr = cline.substring(i-boundaries[i]+1, i+1);
        segstrRightBuffer.addFirst(substr);
      }
    }

    return segstrRightBuffer;
  } 

  private static String concatLineLeft(String cline, int [] boundaries, String separator) {
    int i;

    StringBuffer clinebuffer = new StringBuffer();

    for(i=0; i<boundaries.length; i++) {
      if(boundaries[i] > 0)
      {
        clinebuffer.append(cline.substring(i, i+boundaries[i]));
        clinebuffer.append(separator);
      }
    }
   
    return clinebuffer.toString();
  }

  private static String concatLineRight(String cline, int [] boundaries, String separator) {
    int i;
    String substr;

    StringBuffer clinebuffer = new StringBuffer();

    for(i = boundaries.length-1; i >= 0; i--) {
      if(boundaries[i] > 0 && i-boundaries[i]+1 >= 0)
      {
        substr = cline.substring(i-boundaries[i]+1, i+1);
        clinebuffer.insert(0, substr);
        clinebuffer.insert(substr.length(), separator);
      }
    }

    return clinebuffer.toString();
  } 

  private void segmentLineLeftOffsets(String cline, int[] offsets) {
    int i, j, tmpoffset;
    int clength = cline.length();

    i = 0;
    while (i < clength) {
      if(i+3 <= clength && cline.substring(i, i+3).compareTo("<s>")==0) {
        offsets[i] = 3;
        i += 3;
        continue;
      }

      if(i+4 <= clength && cline.substring(i, i+4).compareTo("</s>")==0) {
        offsets[i] = 4;
        i += 4;
        continue;
      }

      if (Character.UnicodeBlock.of(cline.charAt(i)) == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) {
        j = maxword_length;
        if (i+j > clength) { j = clength - i; }
        while(i+j <= clength && j > 1) {
          if (wordprob_mapdata.containsKey(cline.substring(i, i+j))) break; 
          j--;
        }
        offsets[i] = j;
        i += j;
      } else if (Character.isWhitespace(cline.charAt(i))) {
          j=1;
          while ( i+j < clength && Character.isWhitespace(cline.charAt(i+j)) && (Character.UnicodeBlock.of(cline.charAt(i+j)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ) {
            j++;
          }
          i += j;
      } else if (Character.isLetter(cline.charAt(i))) {
          j=1;
          while( i+j < clength && Character.isLetter(cline.charAt(i+j)) && (Character.UnicodeBlock.of(cline.charAt(i+j)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ) {
            j++;
          }
          offsets[i] = j;
          i += j;
      } else if (Character.isDigit(cline.charAt(i))) {
          j=1;
          while( i+j < clength && (Character.isDigit(cline.charAt(i+j)) || cline.charAt(i+j)=='.') && (Character.UnicodeBlock.of(cline.charAt(i+j)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ){
            j++;
          }
          offsets[i] = j;
          i += j;
      }
      else {
           offsets[i] = 1;
           i++;
      }
    }

    i = 0;
    while (i < clength) {
      if (offsets[i] > 0)   {
        while( i+offsets[i] < clength && offsets[i+offsets[i]] > 0 && i+offsets[i]+offsets[i+offsets[i]] <= clength && isNumber(cline.substring(i, i+offsets[i]+offsets[i+offsets[i]])) ) {
          tmpoffset = offsets[i+offsets[i]];
          offsets[i+offsets[i]] = 0;
          offsets[i] = offsets[i] + tmpoffset;
        }
      }
      i++;
    }

    return;
  }

  private void segmentLineRightOffsets(String cline, int[] offsets) {
    int i, j, k, tmpoffset;
    int clength = cline.length();

    i = clength;
    while (i > 0) {
      if(i-3 > -1 && cline.substring(i-3, i).compareTo("<s>")==0) {
        offsets[i-1] = 3;
        i -= 3;
        continue;
      }

      if(i-4 > -1 && cline.substring(i-4, i).compareTo("</s>")==0) {
        offsets[i-1] = 4;
        i -= 4;
        continue;
      }

      if (Character.UnicodeBlock.of(cline.charAt(i-1)) == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) {
        j = maxword_length;
        if (i-j < 0) { j = i; }
        while(j > 1) {
          if (wordprob_mapdata.containsKey(cline.substring(i-j, i))) break; 
          j--;
        }
        offsets[i-1] = j;
        i -= j;
      } else if (Character.isWhitespace(cline.charAt(i-1))) {
          j=1;
          k = i-j-1;
          while( (k>=0 && Character.isWhitespace(cline.charAt(k))) && (Character.UnicodeBlock.of(cline.charAt(k)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ) {
            j++;
            k = i-j-1;
          }
          i -= j;
      } else if (Character.isLetter(cline.charAt(i-1))) {
          j=1;
          k = i-j-1;
          while( (k>=0 && Character.isLetter(cline.charAt(k))) && (Character.UnicodeBlock.of(cline.charAt(k)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) ) {
            j++;
            k = i-j-1;
          }
          offsets[i-1] = j;
          i -= j;
      } else if (Character.isDigit(cline.charAt(i-1))) {
          j=1;
          k = i-j-1;
          while( (k>=0 && Character.isDigit(cline.charAt(k))) && (Character.UnicodeBlock.of(cline.charAt(k)) != Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS) || (k>=0 && cline.charAt(k)=='.') ){
            j++;
            k = i-j-1;
          }
          offsets[i-1] = j;
          i -= j;
      }
      else {
           offsets[i-1] = 1;
           i--;
      }
    }

    i = clength-1;
    while (i > 0) {
      if(offsets[i] > 0)   {
        while(i-offsets[i]+1 > 0 && offsets[i-offsets[i]] > 0 && i-offsets[i]-offsets[i-offsets[i]]+1 >= 0 && isNumber(cline.substring(i-offsets[i]-offsets[i-offsets[i]]+1, i+1))) {
          tmpoffset = offsets[i-offsets[i]];
          offsets[i-offsets[i]] = 0;
          offsets[i] = offsets[i] + tmpoffset;
        }
      }
      i--;
    }

    return;
  }

  public void segmentFile(String inputfile, int mode) {
    String outfile = inputfile + ".seg";
    String segstring;

    try {
      String dataline;
      InputStream in = new FileInputStream(inputfile);
      BufferedReader rd = new BufferedReader(new InputStreamReader(in, encoding));
      BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outfile), encoding));
	    
      dataline = rd.readLine();
      while(dataline != null) {
        segstring = segmentLine(dataline, seg_separator, mode);
        out.write(segstring);
        out.newLine();
        dataline = rd.readLine();
      }

      in.close();
      rd.close();
      out.close();
    }
    catch (Exception e) {
      System.err.println("Exception " + e.toString());
    }

//  System.gc();
  }

  private static int setInputMode(String modeStr) {
    
    if(modeStr.equals("-mode1")) running_mode = 1;        // left 
    else if(modeStr.equals("-mode2")) running_mode = 2;   // right
    else if(modeStr.equals("-mode3")) running_mode = 3;   // left right short
    else if(modeStr.equals("-mode4")) running_mode = 4;   // left right best prob
    else if(modeStr.equals("-mode5")) running_mode = 5;   // viterbi
    else running_mode = 0;
  
    return running_mode;
  }

  public static void printHelp() {
    System.out.println("Usage: java ChiUtf8Segmenter [-mode1|-mode2|-mode3|-mode4|-mode5] input_file wordprob_map [numbers_identity_file]\n");
    System.out.println("Default option:\t\tleft longest segments");
    System.out.println("Option: -mode1\t\tleft longest segments");
    System.out.println("Option: -mode2\t\tright longest segments");
    System.out.println("Option: -mode3\t\tchoose segments from left longest or right longest (which give less segments)");
    System.out.println("Option: -mode4\t\tsegments with higher unigram probability of left longest or right longest");
    System.out.println("Option: -mode5\t\tViterbi search segmentation (by unigram log probability path cost)\n");
    System.out.println("Segmented text will be saved to input_file.seg");
    System.exit(0);
  }

  public static void main(String[] argv) throws IOException {
    int mode;
    String inputfile;
    ChiUtf8Segmenter segmenter = null;
  
    if(argv.length<2 || argv.length>4) {
      printHelp();
      System.exit(0);
    }

    if(argv.length == 2) {
      if(setInputMode(argv[0])!=0) {
        printHelp();
        System.exit(0);
      }
      mode = 1; // default mode
      inputfile = argv[0];
      segmenter = new ChiUtf8Segmenter(argv[1]); // wordprob_map
    }
    else if(argv.length ==4) {
      mode = setInputMode(argv[0]);
      if(mode == 0) {
        printHelp();
        System.exit(0);
      }
      inputfile = argv[1];
      segmenter = new ChiUtf8Segmenter(argv[2], argv[3]); // wordprob_map numbers_idt_file
    }
    else {
      mode = setInputMode(argv[0]);
      if(mode == 0) { // unknown in this case, so we assume no input of mode and use default mode
        mode = 1;
        inputfile = argv[0];
        segmenter = new ChiUtf8Segmenter(argv[1], argv[2]); // wordprob_map numbers_idt_file
      }   
      else {
        inputfile = argv[1];
        segmenter = new ChiUtf8Segmenter(argv[2]); // wordprob_map
      }
    }

    System.out.println("Total keys " + segmenter.wordprob_mapdata.size());
    segmenter.segmentFile(inputfile, mode);
    System.out.println("Segmentation finished, " + inputfile + " => " + inputfile + ".seg\n");
    segmenter.cleanRes();
  }
}
