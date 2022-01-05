package net.seninp.grammarviz.model;

/**
 * A unified data structure for passing messages between MVC components.
 * 
 * @author psenin
 * 
 */
public class GrammarVizMessage {

  /** The data file name message key. */
  public static final String DATA_FNAME = "data_file_name";

  /** The status message key, this is used for logging. */
  public static final String STATUS_MESSAGE = "status_message";

  /** The processed data message key. (need to rename). */
  public static final String CHART_MESSAGE = "chart_message";

  /** The time-series data message. */
  public static final String TIME_SERIES_MESSAGE = "time_series_message";

  /** The data file name message key. */
  public static final String MAIN_CHART_CLICKED_MESSAGE = "main_chart_clicked_message";

  /** The key storage. */
  private String type;

  /** The payload. */
  private Object payload;

  /**
   * Constructor.
   * 
   * @param messageType set the message type.
   * @param payload set the payload.
   */
  public GrammarVizMessage(String messageType, Object payload) {
    this.type = messageType;
    this.payload = payload;
  }

  /**
   * Retrieve the message payload.
   * 
   * @return the payload.
   */
  public Object getPayload() {
    return payload;
  }

  /**
   * Set the message payload.
   * 
   * @param payload the payload.
   */
  public void setPayload(Object payload) {
    this.payload = payload;
  }

  /**
   * Get the message type.
   * 
   * @return message type.
   */
  public String getType() {
    return type;
  }

  /**
   * Set the message type.
   * 
   * @param type message type.
   */
  public void setType(String type) {
    this.type = type;
  }

}
