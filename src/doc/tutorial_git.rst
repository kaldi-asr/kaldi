====================================================
Kaldi Tutorial: Version control with Git (5 minutes)
====================================================

------------
Introduction
------------

Git is a distributed version control system. This means that, unlike Subversion, there are multiple copies of the repository, and the changes are transferred between these copies in multiple different ways explicitly, but most of the time one's work is backed by a single copy of the repository. Because of this multiplicity of copies, there are multiple possible *workflows* that you may want to follow. Here's one we think best suits you if you just want to *compile and use* Kaldi at first, but then at some point optionally decide to *contribute* your work back to the project.

First-time Git setupIf you have never used Git before, `perform some minimal configuration first <https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup>`_. At the very least, set up your name and e-mail address:

.. code-block:: bash

  $ git config --global user.name "John Doe"
  $ git config --global user.email johndoe@example.com

Also, set short names for the most useful git commands you type most often.

.. code-block:: bash

  $ git config --global alias.co checkout
  $ git config --global alias.br branch
  $ git config --global alias.st status

Another very useful utility comes with ``git-prompts.sh``\ , a bash prompt extension utility for Git (if you do not have it, search the internet how to install it on your system). When installed, it provides a shell function ``__git_ps1`` that, when added to the prompt, expands into the current branch name and pending commit markers, so you do not forget where you are. You may modify your ``PS1`` shell variable so that it includes literally ``$(__git_ps1 "[%s]")``. I have this in my ``~/.bashrc``:

.. code-block:: bash

   PS1='\[\033[00;32m\]\u@\h\[\033[0m\]:\[\033[00;33m\]\w\[\033[01;36m\]$(__git_ps1 "[%s]")\[\033[01;33m\]\$\[\033[00m\] '
   export GIT_PS1_SHOWDIRTYSTATE=true GIT_PS1_SHOWSTASHSTATE=true
   # fake __git_ps1 when git-prompts.sh not installed
   if [ "$(type -t __git_ps1)" == "" ]; then
     function __git_ps1() { :; }
   fi

-----------------
The User Workflow
-----------------

Set up your repository and the working directory with this command:

.. code-block:: bash

  kkm@yupana:~$ git clone https://github.com/kaldi-asr/kaldi.git --branch master --single-branch --origin golden
  Cloning into 'kaldi'...
  remote: Counting objects: 51770, done.
  remote: Compressing objects: 100% (8/8), done.
  remote: Total 51770 (delta 2), reused 0 (delta 0), pack-reused 51762
  Receiving objects: 100% (51770/51770), 67.72 MiB | 6.52 MiB/s, done.
  Resolving deltas: 100% (41117/41117), done.
  Checking connectivity... done.
  kkm@yupana:~$ cd kaldi/
  kkm@yupana:~/kaldi[master]$

Now, you are ready to configure and compile Kaldi and work with it. Once in a while you want the latest changes in your local branch. This is akin to what you usually did with ``svn update``.

But please first let's agree to one thing: you do not commit any files on the master branch. We'll get to that below. So far, you are only using the code. It will be hard to untangle if you do not follow the rule, and Git is so amazingly easy at branching, that you always want to do your work on a branch.

.. code-block:: bash

  kkm@yupana:~/kaldi[master]$ git pull golden
  remote: Counting objects: 148, done.
  remote: Compressing objects: 100% (55/55), done.
  remote: Total 148 (delta 111), reused 130 (delta 93), pack-reused 0
  Receiving objects: 100% (148/148), 18.39 KiB | 0 bytes/s, done.
  Resolving deltas: 100% (111/111), completed with 63 local objects.
  From https://github.com/kaldi-asr/kaldi
     658e1b4..827a5d6  master     -> golden/master

The command you use is ``git pull``\ , and ``golden`` is the alias we used to designate the main replica of the Kaldi repository before.


------------------------
From User To Contributor
------------------------
At some point you decided to change Kaldi code, be it scripts or source. Maybe you made a simple bug fix. Maybe you are contributing a whole recipe. In any case, your always do your work on a branch. Even if you have uncommitted changes, Git handles that. For example, you just realized that the ``fisher_english`` recipe does not actually make use of ``hubscr.pl`` for scoring, but checks it exists and fails. You quickly fixed that in your work tree and want to share this change with the project.


Work locally on a branch

.. code-block:: bash

  kkm@yupana:~/kaldi[master *]$ git fetch golden
  kkm@yupana:~/kaldi[master *\ ]$ git co golden/master -b fishfix --no-track
  M       fisher_english/s5/local/score.sh
  Branch fishfix set up to track remote branch master from golden.
  Switched to a new branch 'fishfix'
  kkm@yupana:~/kaldi[myfix *]$

So what we did here, we first *fetched* the current changes to the golden repository to your machine. This did not update your master (in fact, you cannot pull if you have local worktree changes), but did update the remote reference ``golden/master``. In the second command, we forked off a branch in your local repository, called ``fishfix``. Was it more logical to branch off ``master``\ ? Not at all! First, this is one operation more. You do not *need* to update the master, so why would you? Second, we agreed (remember?) that master will have no changes, and you had some. Third, and believe me, this happens, you might have committed something to your master by mistake, and you do not want to bring this feral change into your new branch.

Now you examine your changes, and, since they are good, you commit them:

.. code-block:: diff

   kkm@yupana:~/kaldi[fishfix *]$ git diff
   diff --git a/egs/fisher_english/s5/local/score.sh b/egs/fisher_english/s5/local/score.sh
   index 60e4706..552fada 100755
   --- a/egs/fisher_english/s5/local/score.sh
   +++ b/egs/fisher_english/s5/local/score.sh
   @@ -27,10 +27,6 @@ dir=$3

    model=$dir/../final.mdl # assume model one level up from decoding dir.

   -hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
   -[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
   -hubdir=`dirname $hubscr`
   -
    for f in $data/text $lang/words.txt $dir/lat.1.gz; do
      [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
    done

.. code-block:: bash

   kkm@yupana:~/kaldi[fishfix *]$ git commit -am 'fisher_english scoring does not really need hubscr.pl from sctk.'
   [fishfix d7d76fe] fisher_english scoring does not really need hubscr.pl from sctk.
    1 file changed, 4 deletions(-)
   kkm@yupana:~/kaldi[fishfix]$

Note that the ``-a`` switch to ``git commit`` makes it commit all modified files (we had only one changed, so why not?). If you want to separate file modifications into multiple features to submit separately, ``git add`` specific files followed by ``git commit`` without the ``-a`` switch, and then start another branch off the same point as the first one for the next fix: ``git co golden/master -b another-fix no-track``\ , where you could add and commit other changed files. With Git, it is not uncommon to have a dozen branches going. Remember that it is extremely easy to combine multiple feature branches into one, but splitting one large changeset into many smaller features involves more work.

Now you need to create a pull request to the maintaners of Kaldi, so that they can pull the change from your repository. For that, *your repository* needs to be available online to them. And for that, you need a GitHub account.

---------------------
One-time GitHub setup
---------------------


*  Go to `main Kaldi repository page <https://github.com/kaldi-asr/kaldi>`_ and click on the Fork button. If you do not have an account, GitHub will lead you through necessary steps. 

*  `Generate and register an SSH key <https://help.github.com/articles/generating-ssh-keys/>`_ with GitHub so that GitHub can identify you. Everyone can read everything on GitHub, but only you can write to your forked repository!

-----------------------
Creating a pull request
-----------------------
Make sure your fork is registered under the name ``origin`` (the alias is arbitrary, this is what we'll use here). If not, add it. The URL is listed on your repository page under "SSH clone URL", and looks like ``[git@github.com](mailto:git@github.com):YOUR_USER_NAME/kaldi.git``.

.. code-block:: bash

  kkm@yupana:~/kaldi[fishfix]$ git remote -v
  golden  https://github.com/kaldi-asr/kaldi.git (fetch)
  golden  https://github.com/kaldi-asr/kaldi.git (push)
  kkm@yupana:~/kaldi[fishfix]$ git remote add origin git@github.com:kkm000/kaldi.git
  kkm@yupana:~/kaldi[fishfix]$ git remote -v
  golden  https://github.com/kaldi-asr/kaldi.git (fetch)
  golden  https://github.com/kaldi-asr/kaldi.git (push)
  origin  git@github.com:kkm000/kaldi.git (fetch)
  origin  git@github.com:kkm000/kaldi.git (push)

Now push the branch into your fork of Kaldi:

.. code-block:: bash

  kkm@yupana:~/kaldi[fishfix]$ git push origin HEAD -u
  Counting objects: 632, done.
  Delta compression using up to 12 threads.
  Compressing objects: 100% (153/153), done.
  Writing objects: 100% (415/415), 94.45 KiB | 0 bytes/s, done.
  Total 415 (delta 324), reused 326 (delta 262)
  To git@github.com:kkm000/kaldi.git


  * [new branch]      HEAD -> fishfix
    Branch fishfix set up to track remote branch fishfix from origin.

``HEAD`` in ``git push`` tells Git "create branch in the remote repo with
the same name as the current branch", and ``-u`` remembers the connection between your local branch ``fishfix`` and ``origin/fishfix`` in your repository.

Now go to your repository page and `create a pull request <https://help.github.com/articles/creating-a-pull-request/>`_. `Examine your changes <https://github.com/kaldi-asr/kaldi/pull/31>`_\ , and submit the request if everything looks good. The maintainers will receive the request and either accept it or comment on it. Follow the comments, commit fixes on your branch, push to ``origin`` again, and GitHub will automatically update the pull request web page. Then reply e. g. "Done" under the comments that you received, so that they know you followed up on their comments.

If you are creating a pull request only for a review of an incomplete piece of work, which makes sense and is encouraged if you want early feedback on a proposed feature, begin the title of your pull request with the prefix ``WIP:``. This will tell the maintainers not to merge the pull request yet. When you push more commits to your branch, they automatically show in the pull request. When you think the work is complete, edit the pull request title to remove the ``WIP`` prefix and then add a comment to this effect, so that the maintainers are notified.

