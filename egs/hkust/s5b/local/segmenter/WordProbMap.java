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

import java.io.*;
import java.lang.*;
import java.util.*;
import java.util.regex.*;

// class for wordlist and corresponding log probabilities(or cost in negative values for segmentation) 
class WordProbMap {
  private String mapName = "wordprobmap";
  private String encoding = "UTF-8";
  private HashMap<String, Float> probmap = null;

  public void setName(String mapName) { this.mapName = mapName; }
  public void setEncoding(String encoding) { this.encoding = encoding; }
  public String getName() { return mapName; }
  public String getEncoding() { return encoding; }
  public HashMap<String, Float> getProbMap() { return probmap; }
  
  public WordProbMap() throws IOException {
    if(readWordProbMap()==false) throw new IOException("read wordprobmap error in WordProbMap.java\n"); 
  }
  
  public WordProbMap(String wordMapFile, String encoding) throws IOException {
    setName(wordMapFile);
    setEncoding(encoding);
    if(readWordProbMap()==false) throw new IOException("read wordprobmap: " + wordMapFile + " error in WordProbMap.java\n");
  }

  public void clearMap() {
    if(probmap != null) {
      probmap.clear();
      probmap = null;
    }
  }

  private boolean readWordProbMap() {
    try {
      FileInputStream fin = new FileInputStream(mapName);
      BufferedReader rd = new BufferedReader(new InputStreamReader(fin, encoding));
      probmap = new HashMap<String, Float>();

      Pattern p = Pattern.compile("[ \t\r\n]+");
      String [] b;
      int line_num = 0;

      String a = rd.readLine();
      while(a != null) {
        line_num++;
        b = p.split(a);
        if(b.length == 0) {
          continue;	// empty line
        }
        else if(b.length != 2) {
          throw new IOException("read wordprobmap: "+mapName+" error in line "+line_num+"\n");
        }
        if(probmap.containsKey(b[0]) && probmap.get(b[0])>Float.valueOf(b[1]) ) { // appear multiple times, choose max
          a = rd.readLine();
          continue;
        }
        probmap.put(b[0], Float.valueOf(b[1]));
        a = rd.readLine();
      }
      fin.close();
      rd.close();
    }
    catch (IOException e) {
      System.err.println(e);
      return false;
    }

    return true;
  }
}

