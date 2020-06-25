#!/usr/bin/env python
#
# Copyright (c) 2013 Tanel Alumae
#
# Slightly inspired by the CMU Sphinx's Pocketsphinx Gstreamer plugin demo (which has BSD license)
#
# Apache 2.0

from __future__ import print_function
import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, Gtk, Gdk
GObject.threads_init()
Gdk.threads_init()
Gst.init(None)

class DemoApp(object):
    """GStreamer/Kaldi Demo Application"""
    def __init__(self):
        """Initialize a DemoApp object"""
        self.init_gui()
        self.init_gst()

    def init_gui(self):
        """Initialize the GUI components"""
        self.window = Gtk.Window()
        self.window.connect("destroy", self.quit)
        self.window.set_default_size(400,200)
        self.window.set_border_width(10)
        vbox = Gtk.VBox()        
        self.text = Gtk.TextView()
        self.textbuf = self.text.get_buffer()
        self.text.set_wrap_mode(Gtk.WrapMode.WORD)
        vbox.pack_start(self.text, True, True, 1)
        self.button = Gtk.Button("Speak")
        self.button.connect('clicked', self.button_clicked)
        vbox.pack_start(self.button, False, False, 5)
        self.window.add(vbox)
        self.window.show_all()

    def quit(self, window):
        Gtk.main_quit()

    def init_gst(self):
        """Initialize the speech components"""
        self.pulsesrc = Gst.ElementFactory.make("pulsesrc", "pulsesrc")
        if self.pulsesrc == None:
            print("Error loading pulsesrc GST plugin. You probably need the gstreamer1.0-pulseaudio package", file=sys.stderr)
            sys.exit()	
        self.audioconvert = Gst.ElementFactory.make("audioconvert", "audioconvert")
        self.audioresample = Gst.ElementFactory.make("audioresample", "audioresample")    
        self.asr = Gst.ElementFactory.make("onlinegmmdecodefaster", "asr")
        self.fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
        
        if self.asr:
          model_dir = "online-data/models/tri2b_mmi/"
          if not os.path.isdir(model_dir):
              print("Model (%s) not downloaded. Run run-simulated.sh first" % model_dir, file=sys.stderr)
              sys.exit(1)
          self.asr.set_property("fst", model_dir + "HCLG.fst")
          self.asr.set_property("lda-mat", model_dir + "matrix")
          self.asr.set_property("model", model_dir + "model")
          self.asr.set_property("word-syms", model_dir + "words.txt")
          self.asr.set_property("silence-phones", "1:2:3:4:5")
          self.asr.set_property("max-active", 4000)
          self.asr.set_property("beam", 12.0)
          self.asr.set_property("acoustic-scale", 0.0769)
        else:
          print("Couldn't create the onlinegmmfasterdecoder element. ", file=sys.stderr)
          if "GST_PLUGIN_PATH" in os.environ:
            print("Have you compiled the Kaldi GStreamer plugin?", file=sys.stderr)
          else:
            print("You probably need to set the GST_PLUGIN_PATH envoronment variable", file=sys.stderr)
            print("Try running: GST_PLUGIN_PATH=../../../src/gst-plugin %s" % sys.argv[0], file=sys.stderr)
          sys.exit();
        
        # initially silence the decoder
        self.asr.set_property("silent", True)
        
        self.pipeline = Gst.Pipeline()
        for element in [self.pulsesrc, self.audioconvert, self.audioresample, self.asr, self.fakesink]:
            self.pipeline.add(element)         
        self.pulsesrc.link(self.audioconvert)
        self.audioconvert.link(self.audioresample)
        self.audioresample.link(self.asr)
        self.asr.link(self.fakesink)    
  
        self.asr.connect('hyp-word', self._on_word)
        self.pipeline.set_state(Gst.State.PLAYING)


    def _on_word(self, asr, word):
        Gdk.threads_enter()
        if word == "<#s>":
          self.textbuf.insert_at_cursor("\n")
        else:
          self.textbuf.insert_at_cursor(word)
        self.textbuf.insert_at_cursor(" ")
        Gdk.threads_leave()

    def button_clicked(self, button):
        """Handle button presses."""
        if button.get_label() == "Speak":
            button.set_label("Stop")
            self.asr.set_property("silent", False)
        else:
            button.set_label("Speak")
            self.asr.set_property("silent", True)
            

if __name__ == '__main__':
  app = DemoApp()
  print('''
  The (bigram) language model used to build the decoding graph was
  estimated on an audio book's text. The text in question is
  King Solomon's Mines" (http://www.gutenberg.org/ebooks/2166).
  You may want to read some sentences from this book first ...''')

  Gtk.main()
