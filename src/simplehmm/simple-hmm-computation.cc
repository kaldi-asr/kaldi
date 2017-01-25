SimpleHmmComputation::SimpleHmmComputation(
    const SimpleHmm &model, 
    const std::vector<int32> &num_pdfs, 
    VectorFst<StdArc> *decode_fst, 
    const Matrix<BaseFloat> &log_likes)
