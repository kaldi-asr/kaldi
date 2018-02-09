// idlaktxp/convert-tree-pdf-hts.cc

// Copyright 2014 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#include "idlaktxpbin/convert-tree-pdf-hts.h"

namespace kaldi {
void _convert_tree(const std::string &tab, const EventMap * emap,
                   bool yes,
                   const ConvertCexVector &cvncex_array,
                   TreeNodeVector * treenode_array,
                   LookupInt * contextname2index,
                   int32  state, TreeNode * parent);
// void _readqsets(std::istream &is, std::istream &htsis, LookupMap * lkp);
void _read_context_setup(std::string kaldicontextsetup_filename,
                         int32 N,
                         const fst::SymbolTable &phones_symtab,
                         ConvertCexVector * cvncex_array);
}
// This takes tree and gmm based model output from the idlak build system
// together with HTS question set definitions and writes pdfs and trees
// which are compatible with hts engine.

int main(int argc, char *argv[]) {
  using namespace kaldi;
  ConvertCexVector cvncex_array;
  // lookup to ensure question names generated for hts are unique
  LookupInt contextname2index;
  kaldi::int32 N = 5;
  kaldi::int32 num_states = 5;
  bool pitchdata = false;
  kaldi::int32 stateno;
  kaldi::int32 nodeno;
  kaldi::int32 leafno = 0;
  kaldi::int32 totleaves = 0;
  kaldi::int32 totmodels = 0;
  const char *usage =
      "Convert kaldi tree and gmm based model to an HTS format\n"
      "Usage: convert-tree-pdf-hts [options] kaldiphoneset kaldicontextsetup tree model htstreeout htsmodelout\n";
  // input output variables
  std::string tree_in_filename;
  std::string model_in_filename;
  std::string tree_out_filename;
  std::string model_out_filename;
  std::string phones_symtab_filename;
  std::string kaldicontextsetup_filename;
  std::string kaldiqset_filename;
  std::string htsqset_filename;
  std::string tab;
  ContextDependency ctx_dep;
  TransitionModel trans_model;
  AmDiagGmm am_gmm;
  const DiagGmm * pdf;
  StringVector kaldivals;
  kaldi::int32 i, j;
  kaldi::Vector<BaseFloat> weights;
  kaldi::Matrix<BaseFloat> means;
  kaldi::Matrix<BaseFloat> vars;
  try {
    kaldi::ParseOptions po(usage);
    po.Register("context-width", &N, "Context window size");
    po.Register("number-states", &num_states, "Number of states in hmm topology");
    // add option to produce 'streamed lf0'
    po.Register("pitch", &pitchdata, "Indicates this is pitch data: The first parameter is interpreted as probability of voicing and output as a stream weight for HTS. ");
    // how do we write to two files in kaldi? Can't stream what so waht is the etiquette?
    po.Read(argc, argv);
    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }
    phones_symtab_filename = po.GetArg(1);
    kaldicontextsetup_filename = po.GetArg(2);
    tree_in_filename = po.GetArg(3);
    model_in_filename = po.GetArg(4);
    tree_out_filename = po.GetArg(5);
    model_out_filename = po.GetArg(6);
    kaldi::Output treeout(tree_out_filename, false); // not binary
    std::filebuf fb;
    // Can't use kaldi to open the stream because it adds a header code when its in binary
    fb.open(model_out_filename.c_str(),std::ios::out);
    std::ostream os(&fb);
    //kaldi::Output modelout(model_out_filename, true); // binary

    // read the kaldi phone to integer mapping
    fst::SymbolTable *phones_symtab = NULL;
    {
      std::ifstream is(phones_symtab_filename.c_str());
      phones_symtab = fst::SymbolTable::ReadText(is, phones_symtab_filename);
      if (!phones_symtab || phones_symtab->NumSymbols() == 0)
        KALDI_ERR << "Error opening symbol table file "<<phones_symtab_filename;
    }
    _read_context_setup(kaldicontextsetup_filename, N,
                        *phones_symtab,
                        &cvncex_array);
    
    ReadKaldiObject(tree_in_filename, &ctx_dep);
    bool binary_read;
    Input ki(model_in_filename, &binary_read);
    trans_model.Read(ki.Stream(), binary_read);
    am_gmm.Read(ki.Stream(), binary_read);
    for (i = 0; i < am_gmm.NumPdfs(); i++) {
      pdf = &am_gmm.GetPdf(i);
      // only one guassian allowed
      if (pdf->NumGauss() != 1) {
        KALDI_ERR << "Model as more than 1 Guassian (" << pdf->NumGauss() << ")";
      }
      weights = pdf->weights();
      // output weights
      //std::cout << weights(0) << " ";
      pdf->GetMeans(&means);
      pdf->GetVars(&vars);
      // output means
      //std::cout << "MEANS: ";
      for (j = 0; j < pdf->Dim(); j++) {
        //std::cout << means(0, j) << " ";
        //if (j < pdf->Dim() - 1) std::cout << " ";
        //else std::cout << "\n";
      }
      // output variances
      //std::cout << "VARS: ";
      for (j = 0; j < pdf->Dim(); j++) {
        //std::cout << vars(0, j);
        //if (j < pdf->Dim() - 1) std::cout << " ";
        //else std::cout << "\n";
      }
    }    
    // Input htsqset;
    // htsqset.OpenTextMode(htsqset_filename);
    // Input kaldiqset;
    // kaldiqset.OpenTextMode(kaldiqset_filename);
    // _readqsets(kaldiqset.Stream(), htsqset.Stream(), &kaldi2htsqset);
    std::vector<TreeNodeVector> treenode_array_array;
    for (kaldi::int32 s = 0; s < num_states; s++) {
      // write zeros to fill in later for number of leafs per state
      os.write((const char *)(&leafno), sizeof(leafno));
      TreeNodeVector treenode_array;
      _convert_tree(tab, &ctx_dep.ToPdfMap(), false, cvncex_array, &treenode_array,
                   &contextname2index, s, NULL);
      //output questions
      for(TreeNodeVector::iterator iter = treenode_array.begin();
          iter != treenode_array.end();
          iter++) {
        treeout.Stream() << "QS " << (*iter)->GetHtsQuestion() << "\n";
      }
      treenode_array_array.push_back(treenode_array);
    }
    treeout.Stream() << "\n";
    stateno = 2;
    for (std::vector<TreeNodeVector>::iterator iterv = treenode_array_array.begin();
         iterv != treenode_array_array.end();
         stateno++, iterv++) {    
      treeout.Stream() << "{*}[" << stateno << "]\n{\n";
      nodeno = 0;
      leafno = 1;
      totleaves = 0;
      for(TreeNodeVector::iterator iter = iterv->begin();
          iter != iterv->end();
          nodeno++, iter++) {
        //output tree
        if (nodeno) treeout.Stream() << "-";
        treeout.Stream() << nodeno << " " << (*iter)->GetHtsName() << " ";
        //no value
        if ((*iter)->IsTerminalNo()) {
          treeout.Stream() << "lf0_s" << stateno << "_" << leafno << " ";
          //output the pdf
          pdf = &am_gmm.GetPdf((*iter)->GetNo());
          pdf->GetMeans(&means);
          pdf->GetVars(&vars);
          if (pitchdata) j = 1;
          else j = 0;             
          for (; j < pdf->Dim(); j++) {
            float m = means(0, j);
            os.write((const char *)&m, sizeof(m));
            //if (leafno > 205 && leafno < 227)
            //  cout << leafno << "MEAN " << m << "\n";
          }
          if (pitchdata) j = 1;
          else j = 0;             
          for (; j < pdf->Dim(); j++) {
            float v = vars(0, j);
            os.write((const char *)&v, sizeof(v));
            //if (leafno > 205 && leafno < 227)
            //  cout << leafno << "VAR " << v << "\n";
          }
          if (pitchdata) {
            float w = means(0, 0);
            os.write((const char *)&w, sizeof(w));
            //if (leafno > 205 && leafno < 227)
            //  cout << leafno << "WEIGHT " << w << "\n";
          }
          totmodels++;
          totleaves++;
          leafno++;
        }
        else {
          treeout.Stream() << "-" << (*iter)->GetNo() << " ";
        }
        //yes value
        if ((*iter)->IsTerminalYes()) {
          treeout.Stream() << "lf0_s" << stateno << "_" << leafno << "\n";
          //output the pdf
          pdf = &am_gmm.GetPdf((*iter)->GetYes());
          pdf->GetMeans(&means);
          pdf->GetVars(&vars);
          if (pitchdata) j = 1;
          else j = 0;             
          for (; j < pdf->Dim(); j++) {
            float m = means(0, j);
            os.write((const char *)&m, sizeof(m));
            //if (leafno > 205 && leafno < 227)
            //  cout << leafno << "MEAN " << m << "\n";
          }
          if (pitchdata) j = 1;
          else j = 0;             
          for (; j < pdf->Dim(); j++) {
            float v = vars(0, j);
            os.write((const char *)&v, sizeof(v));
            //if (leafno > 205 && leafno < 227)
            //  cout << leafno << "VAR " << v << "\n";
          }
          if (pitchdata) {
            float w = means(0, 0);
            os.write((const char *)&w, sizeof(w));
            //if (leafno > 205 && leafno < 227)
            //  cout << leafno << "WEIGHT " << w << "\n";
          }
          totmodels++;
          totleaves++;
          leafno++;
        }
        else {
          treeout.Stream() << "-" << (*iter)->GetYes() << "\n";
        }
      }
      treeout.Stream() << "}\n\n";
      std::cout << stateno << " " << leafno << "\n";
      os.flush();
      os.seekp((stateno - 2) * 4, std::ios_base::beg);
      os.write((const char *)(&totleaves), sizeof(leafno));
      os.seekp(0, std::ios_base::end);
    }
    fb.close();
    std::cout << "TOT:" << totmodels << "\n";   
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

namespace kaldi {

// read kaldi context setup xml to allow on th fly question generation from
// new questions added by kaldi i.e. by merging questions or for built in
// kaldi phone questions.
void _read_context_setup(std::string kaldicontextsetup_filename,
                         int32 N,
                         const fst::SymbolTable &phones_symtab,
                         ConvertCexVector * cvncex_array) {
  int32 i, j, pos;
  bool kphonefound = false, binary;
  // Read XMl context setup information
  Input ki(kaldicontextsetup_filename, &binary);
  pugi::xml_node kphone;
  pugi::xml_node cexfunction;
  pugi::xml_document doc;
  pugi::xml_parse_result r = doc.load(ki.Stream(), pugi::encoding_utf8);
  ConvertCex cnvcex;
  
  // Add this look up for kaldi phones
  pugi::xpath_node_set kphones =
      doc.document_element().select_nodes("//kphone");
  kphones.sort();

  for(i = 0; i < N; i++) {
    pos = i - (N/2);
    std::string name = "";
    // construct a context name
    if (pos < 0) {
      for(j = 0; j > pos; j--)
        name = name + "Backward";
    }
    else if (pos > 0) {
      for(j = 0; j < pos; j++)
        name = name + "Forward";
    }
    name = name + "PhoneKaldi";
    kphonefound = false;
    for (pugi::xpath_node_set::const_iterator it = kphones.begin();
         it != kphones.end();
         ++it) {
     kphone = (*it).node();
     if (!strcmp(kphone.attribute("name").as_string(), name.c_str())) {
       kphonefound = true;
       break;
     }
    }
    if (!kphonefound)
      KALDI_ERR << "No appropriate kphone context in HTS model for: "<< name;

    cnvcex.Init(name,
                std::string(
                    kphone.attribute("htsprv").as_string()),
                std::string(
                    kphone.attribute("htspst").as_string()),
                false);
    // Take symbol names from kaldi phone set and add to the
    // conversion table
    // kaldi phones are by default the first N fields in the kaldi full
    // context integer array
    for (j = 0; j < phones_symtab.NumSymbols(); j++)
      cnvcex.AddVal(phones_symtab.Find(j));
    cvncex_array->push_back(cnvcex);
  }

  // Add the other lookups from the context setup
  pugi::xpath_node_set cexfunctions = doc.document_element().select_nodes("//cexfunction");
  cexfunctions.sort();
  for (pugi::xpath_node_set::const_iterator it = cexfunctions.begin();
       it != cexfunctions.end();
       ++it) {
    int32 isinteger;
    cexfunction = (*it).node();
    isinteger = cexfunction.attribute("isinteger").as_int();
    //std::cout << cexfunction.attribute("name").as_string() << isinteger << "\n";
    cnvcex.Init(std::string(
        cexfunction.attribute("name").as_string()),
                std::string(
                    cexfunction.attribute("htsprv").as_string()),
                std::string(
                    cexfunction.attribute("htspst").as_string()),
                isinteger);
    if (isinteger == 0) {
      pugi::xpath_node_set vals = cexfunction.select_nodes("descendant::val");
      for (pugi::xpath_node_set::const_iterator it = vals.begin();
           it != vals.end();
           ++it) {
        pugi::xml_node val = (*it).node();
        cnvcex.SetVal(val.attribute("mapping").as_int(),
                      std::string(val.attribute("name").as_string()));
      }
    }
    cvncex_array->push_back(cnvcex);
  }
}


// Only questions true for the state value are output
void _convert_tree(const std::string &tab, const EventMap * emap,
                  bool yes,
                  const ConvertCexVector &cvncex_array,
                  TreeNodeVector *treenode_array,
                  LookupInt * contextname2index,
                  int32  state, TreeNode * parent) {
  // to get indentation on recursion
  std::string newtab = tab + "  ";
  // string to hold full kaldi style question
  std::string kaldiq;
  // string to hold hts question name
  std::string htsqname;
  // string to hold full hts context style question
  std::string htsq;
  // tree node
  std::vector<EventMap*> out;
  // set of correct values for yes branch
  const ConstIntegerSet<EventValueType> * yesset;
  // integer to string info to map kaldi question to hts question
  const ConvertCex * cnvcex;
  LookupInt::iterator lkp;
  // if a state question is the answer yes or no
  bool stateqY = false;
  // Get the node information
  emap->GetChildren(&out);
  // It is a non terminal node
  if (out.size()) {
    yesset = emap->YesSet();
    kaldiq.clear();
    std::ostringstream kaldiqstream(kaldiq);
    std::ostringstream htsqstream(kaldiq);
    kaldiqstream << emap->first << " ?";
    // if not a state question retrieve the context info
    if (emap->first != -1) {
      cnvcex = &cvncex_array.at(emap->first);
      htsqstream << cnvcex->GetName();
      lkp = contextname2index->find(cnvcex->GetName());
      if (lkp == contextname2index->end()) {
        contextname2index->insert(LookupIntItem(cnvcex->GetName(), 1));
        htsqstream << "1";
      }
      else {
        contextname2index->at(cnvcex->GetName()) = lkp->second + 1;
        htsqstream << (lkp->second) + 1;
      }
    }
    else
      cnvcex = NULL;
    
    htsqname = htsqstream.str();
    htsqstream.clear();
    htsqstream << " { ";
    for (ConstIntegerSet<EventValueType>::iterator iter = yesset->begin();
         iter != yesset->end();
         ++iter) {
      kaldiqstream << " " << *iter;
      // we do not need to build HTS questiosn for state questions
      // as in HTS these are different trees
      if (emap->first != -1) {
        if (iter != yesset->begin()) htsqstream << ",";
        //std::cout << htsqstream.str() << "!!\n";
        htsqstream << "\"*" << cnvcex->GetPrvdelim();
        if (cnvcex->GetIsinteger()) htsqstream << *iter;
        else htsqstream << cnvcex->LookupVal(*iter);
        htsqstream << cnvcex->GetPstdelim() << "*\"";
      }
      else {
        if (!stateqY && *iter == state) stateqY = true;
      }
      // std::cout << *iter << " ";
    }
    htsqstream << " }";
    //std::cout << "]\n";
    if (emap->first == -1) {
      //std::cout << stateqY << " " <<  state << " " << kaldiqstream.str() << "\n";
      if (stateqY) _convert_tree(tab, out[0], yes, cvncex_array, treenode_array,
                                 contextname2index, state, parent);
      else _convert_tree(tab, out[1], yes, cvncex_array, treenode_array,
                         contextname2index, state, parent);
    }
    else {
      TreeNode * node = new TreeNode(htsqname, htsqstream.str());
      treenode_array->push_back(node);
      if (parent) parent->SetParent(yes, treenode_array->size() - 1);
      //std::cout << newtab << htsqname << "\n";
      //std::cout << newtab << "[" << kaldiqstream.str() << "]\n";
      
      _convert_tree(newtab, out[0], true, cvncex_array, treenode_array,
                    contextname2index, state, node);
      _convert_tree(newtab, out[1], false, cvncex_array, treenode_array,
                    contextname2index, state, node);
    }
  }
  // it is a terminal node
  else {
    int32 val = static_cast<const ConstantEventMap * >(emap)->GetAnswer();
    parent->SetVal(yes, val);
    //std::cout  << newtab << "A:" << static_cast<const ConstantEventMap * >(emap)->GetAnswer() << "\n";
  }
}


}
