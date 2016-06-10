#include <string>
#include "base/timer.h"

using namespace std;
using namespace kaldi;
/**
 *
 * @author hossein hadian
 */
class Profiler {
    typedef pair<string, long> Profile;
public:
    vector<Profile> profiles;
    long tot_dur;

    void tic(string id) {
        if (profiling) {
            tac();
        }
        //cur_profile = new Profile(id, System.currentTimeMillis());
        curr_id = id;
        if (id.length() > max_id_len) {
            max_id_len = id.length();
        }
        profiling = true;
        t.Reset();
    }

    void tac() {
	if(!profiling)
	    return;
        profiling = false;
        long dur = t.Elapsed()*1000;
        tot_dur += dur;
        profiles.push_back(make_pair(curr_id, dur));
    }
    
    string toString(int n, bool percentage) {
        string res = "";
        if (n == 0) {
            n = profiles.size();
        }
        for (unsigned int i = 0; i < profiles.size(); i++) {
            if (n-- <= 0) {
                break;
            }
            string str_second = static_cast<ostringstream*> (&(ostringstream() << profiles[i].second))->str();
            res += profiles[i].first + " ---> " + str_second + " millis\n";
        }
        return res;
    }

    string toString() {
        return toString(0, false);
    }

    Profiler() {
        profiling = false;
        max_id_len = 0;
        tot_dur = 0;
    }
    
private:
    string curr_id;
    int max_id_len;
    kaldi::Timer t;
    bool profiling;
};
