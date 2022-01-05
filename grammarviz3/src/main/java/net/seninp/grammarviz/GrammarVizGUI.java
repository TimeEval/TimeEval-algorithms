package net.seninp.grammarviz;

import java.util.Locale;
import net.seninp.grammarviz.controller.GrammarVizController;
import net.seninp.grammarviz.model.GrammarVizModel;
import net.seninp.grammarviz.view.GrammarVizView;

/**
 * Main runnable of Sequitur GUI.
 * 
 * @author psenin
 * 
 */
public class GrammarVizGUI {

  /** The model instance. */
  private static GrammarVizModel model;

  /** The controller instance. */
  private static GrammarVizController controller;

  /** The view instance. */
  private static GrammarVizView view;

  /**
   * Runnable GIU.
   * 
   * @param args None used.
   */
  public static void main(String[] args) {

    System.out.println("Starting GrammarViz 3.0 ...");

    /** Boilerplate */
    // the locale setup
    Locale defaultLocale = Locale.getDefault();
    Locale newLocale = Locale.US;
    System.out.println(
        "Changing runtime locale setting from " + defaultLocale + " to " + newLocale + " ...");
    Locale.setDefault(newLocale);

    // this is the Apple UI fix
    System.setProperty("apple.laf.useScreenMenuBar", "true");
    System.setProperty("com.apple.mrj.application.apple.menu.about.name", "SAXSequitur");

    /** On the stage. */
    // model...
    model = new GrammarVizModel();

    // controller...
    controller = new GrammarVizController(model);

    // view...
    view = new GrammarVizView(controller);

    // make sure these two met...
    model.addObserver(view);
    controller.addObserver(view);

    // live!!!
    view.showGUI();

  }

}
