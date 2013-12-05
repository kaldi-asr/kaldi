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

import java.lang.*;
import java.util.*;

// class for search history path storage
public class SearchHistoryPath {

  private int number_element;
  private ArrayList<String> element = null;
  private float log_prob;


  public SearchHistoryPath() {
    number_element = 0;
    element = new ArrayList<String>();
    log_prob = 0.0f;
  }

  public void addElement(String strVal, float strProb) {
    number_element++;
    element.add(strVal);
    log_prob+=strProb;    
  }

  public int getNumElement() {
    return number_element;
  }

  public float getLogProb() {
    return log_prob;
  }

  public ArrayList<String> getList() {
    return element;
  }

  public void setList(ArrayList<String> element_path) {
    element.clear();
    ListIterator<String> listIterator = element_path.listIterator();
    while (listIterator.hasNext()) {
      element.add(listIterator.next());
    }
    number_element = element.size();
  }

  public void clear() {
    number_element = 0;
    element.clear();
    element = null;
    log_prob = 0.0f;
  }

}
