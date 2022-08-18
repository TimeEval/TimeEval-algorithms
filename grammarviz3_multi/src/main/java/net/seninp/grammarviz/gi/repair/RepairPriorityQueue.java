package net.seninp.grammarviz.gi.repair;

import net.seninp.grammarviz.gi.repair.RepairDigramRecord;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Implements the priority queue for RePair. Backed by the doubly linked list of custom nodes.
 * 
 * @author psenin
 *
 */
public class RepairPriorityQueue {

  // the head pointer
  private RepairQueueNode head = null;

  // the "quick" pointers <digram string> -> <node>
  private HashMap<String, RepairQueueNode> elements = new HashMap<String, RepairQueueNode>();

  /**
   * Places an element in the queue at the place based on its frequency.
   * 
   * @param digramRecord the digram record to place into.
   */
  public void enqueue(RepairDigramRecord digramRecord) {
    
    // System.out.println("before == " + this.toString());
    // if the same key element is in the queue - something went wrong with tracking...
    if (elements.containsKey(digramRecord.str)) {
      throw new IllegalArgumentException(
          "Element with payload " + digramRecord.str + " already exists in the queue...");
    }
    else {

      // create a new node
      RepairQueueNode nn = new RepairQueueNode(digramRecord);
      // System.out.println(nn.payload);

      // place it into the queue if it's empty
      if (this.elements.isEmpty()) {
        this.head = nn;
      }
      
      // if new node has _higher than_ or _equal to_ the head frequency... this going to be the new head
      else if (nn.getFrequency() >= this.head.getFrequency()) {
        this.head.prev = nn;
        nn.next = this.head;
        this.head = nn;
      }
      
      // in all other cases find an appropriate place in the existing queue, starting from the head
      else {
        
        RepairQueueNode currentNode = head;
        
        while (null != currentNode.next) {
          // the intent is to slide down the list finding a place at new node is greater than a node
          // a tracking pointer points to...
          // ABOVE we just checked that at this loop start that the current node is greater than new
          // node
          //
          if (nn.getFrequency() >= currentNode.getFrequency()) {
            RepairQueueNode prevN = currentNode.prev;
            prevN.next = nn;
            nn.prev = prevN;
            currentNode.prev = nn;
            nn.next = currentNode;
            break; // the element has been placed
          }
          currentNode = currentNode.next;
        }
        
        // check if loop was broken by the TAIL condition, not by placement
        if (null == currentNode.next) {
          // so, currentNode points on the tail...
          if (nn.getFrequency() >= currentNode.getFrequency()) {
            // insert just before...
            RepairQueueNode prevN = currentNode.prev;
            prevN.next = nn;
            nn.prev = prevN;
            currentNode.prev = nn;
            nn.next = currentNode;
          }
          else {
            // or make a new tail
            nn.prev = currentNode;
            currentNode.next = nn;
          }
        }

      }
      // also save the element in the index store
      this.elements.put(nn.payload.str, nn);

    }
    // System.out.println("before == " + this.toString());
  }

  /**
   * Returns the most frequently seen element -- the head of the queue.
   * 
   * @return the digram record from the top of the queue or a null.
   */
  public RepairDigramRecord dequeue() {
    if (null != this.head) {
      RepairDigramRecord el = this.head.payload;
      this.head = this.head.next;
      if (null != this.head) {
        this.head.prev = null;
      }
      this.elements.remove(el.str);
      // System.out.println(this);
      return el;
    }
    return null;
  }

  /**
   * Returns the queue size.
   * 
   * @return the number of elements in the queue.
   */
  public int size() {
    return this.elements.size();
  }

  /**
   * Peaks onto the head element (doesn't remove it).
   * 
   * @return the head element pointer.
   */
  public RepairDigramRecord peek() {
    if (null != this.head) {
      return this.head.payload;
    }
    return null;
  }

  /**
   * Checks if a digram is in the queue.
   * 
   * @param digramStr the digram string.
   * @return true if it is present in the queue.
   */
  public boolean containsDigram(String digramStr) {
    return this.elements.containsKey(digramStr);
  }

  /**
   * Gets an element in the queue given its key.
   * 
   * @param key the key to look for.
   * @return the element which corresponds to the key or null.
   */
  public RepairDigramRecord get(String key) {
    RepairQueueNode el = this.elements.get(key);
    if (null != el) {
      return el.payload;
    }
    return null;
  }

  /**
   * Updates the priority queue according to the change...
   * 
   * @param digram the digram string.
   * @param newFreq new frequency.
   * 
   * @return the pointer onto updated element.
   */
  public RepairDigramRecord updateDigramFrequency(String digram, int newFreq) {

    // if the key exists
    if (!this.elements.containsKey(digram)) {
      return null;
    }

    // get a pointer on that node
    RepairQueueNode alteredNode = elements.get(digram);

    // the trivial case
    if (newFreq == alteredNode.payload.freq) {
      return alteredNode.payload;
    }

    // simply evict the node if the freq is too low
    if (2 > newFreq) {
      removeNodeFromList(alteredNode);
      this.elements.remove(alteredNode.payload.str);
      // System.out.println(this);
      return null;
    }

    // update the frequency
    int oldFreq = alteredNode.payload.freq;
    alteredNode.payload.freq = newFreq;

    // if the list is just too damn short
    if (1 == this.elements.size()) {
      // System.out.println(this);
      return alteredNode.payload;
    }

    // if we have to push the element up in the list
    if (newFreq > oldFreq) {

      // going up here
      RepairQueueNode currentNode = alteredNode.prev;
      if (null == alteredNode.prev) {
        currentNode = alteredNode.next;
      }

      removeNodeFromList(alteredNode);
      alteredNode.next = null;
      alteredNode.prev = null;

      while ((null != currentNode) && (currentNode.payload.freq < alteredNode.payload.freq)) {
        currentNode = currentNode.prev;
      }

      // we hit the head, oops... make it the new head
      if (null == currentNode) {
        alteredNode.next = this.head;
        this.head.prev = alteredNode;
        this.head = alteredNode;
      }
      else {
        if (null == currentNode.next) {
          currentNode.next = alteredNode;
          alteredNode.prev = currentNode;
        }
        else {
          currentNode.next.prev = alteredNode;
          alteredNode.next = currentNode.next;
          currentNode.next = alteredNode;
          alteredNode.prev = currentNode;
        }
      }
    }
    else {

      // what if this is a tail already?
      if (alteredNode.next == null) {
        // System.out.println(this);
        return alteredNode.payload;
      }
      // or if we got to stay in the head
      if (this.head == alteredNode && alteredNode.payload.freq >= this.head.next.payload.freq) {
        // System.out.println(this);
        return alteredNode.payload;
      }

      // going down..
      RepairQueueNode currentNode = alteredNode.next;
      removeNodeFromList(alteredNode);
      alteredNode.next = null;
      alteredNode.prev = null;

      while (null != currentNode.next && currentNode.payload.freq > alteredNode.payload.freq) {
        currentNode = currentNode.next;
      }

      if (null == currentNode.next) { // we hit the tail
        if (alteredNode.payload.freq > currentNode.payload.freq) {
          // place before tail
          if (this.head.equals(currentNode)) {
            alteredNode.next = currentNode;
            currentNode.prev = alteredNode;
            this.head = alteredNode;
          }
          else {
            alteredNode.next = currentNode;
            alteredNode.prev = currentNode.prev;
            currentNode.prev.next = alteredNode;
            currentNode.prev = alteredNode;
          }
        }
        else {
          currentNode.next = alteredNode;
          alteredNode.prev = currentNode;
        }
      }
      else { // place element just before of cp
        alteredNode.next = currentNode;
        alteredNode.prev = currentNode.prev;
        if (null == currentNode.prev) {
          // i.e. we are in da head...
          this.head = alteredNode;
        }
        else {
          currentNode.prev.next = alteredNode;
          currentNode.prev = alteredNode;
        }
      }
    }
    // System.out.println(this);
    return alteredNode.payload;

  }

  /**
   * Needed this for debug purpose -- translates the doubly linked list into an array list.
   * 
   * @return an array list (sorted by priority) of elements (live copy).
   */
  public ArrayList<RepairDigramRecord> toList() {
    ArrayList<RepairDigramRecord> res = new ArrayList<RepairDigramRecord>(this.elements.size());
    RepairQueueNode cp = this.head;
    while (null != cp) {
      res.add(cp.payload);
      cp = cp.next;
    }
    return res;
  }

  /**
   * Removes a node from the doubly linked list which backs the queue.
   * 
   * @param el the element pointer.
   */
  private void removeNodeFromList(RepairQueueNode el) {
    // the head case
    //
    if (null == el.prev) {
      if (null != el.next) {
        this.head = el.next;
        this.head.prev = null;
        el=null;
      }
      else {
        // can't happen? yep. if there is only one element exists...
        this.head = null;
      }
    }
    // the tail case
    //
    else if (null == el.next) {
      if (null != el.prev) {
        el.prev.next = null;
      }
      else {
        // can't happen?
        throw new RuntimeException("Unrecognized situation here...");
      }
    }
    // all others
    //
    else {
      el.prev.next = el.next;
      el.next.prev = el.prev;
    }

  }

  /*
   * (non-Javadoc) Debug message.
   * 
   * @see java.lang.Object#toString()
   */
  public String toString() {
    StringBuffer sb = new StringBuffer("priority queue of ").append(this.elements.size())
        .append(" nodes:\n");
    RepairQueueNode hp = this.head;
    int nodeCounter = 0;
    while (null != hp) {
      sb.append(nodeCounter).append(": ").append(hp.payload.str).append(", ")
          .append(hp.payload.freq);
      sb.append("|");
      if (null == hp.prev) {
        sb.append("null");
      }
      else {
        sb.append("ok");
      }
      sb.append("|");
      if (null == hp.next) {
        sb.append("null");
      }
      else {
        sb.append("ok");
      }
      sb.append("|\n");
      hp = hp.next;
      nodeCounter++;
    }
    return sb.delete(sb.length() - 1, sb.length()).toString();
  }

  /**
   * Implements the repair queue node.
   * 
   * @author psenin
   *
   */
  private class RepairQueueNode {
    // a pointer onto previous node
    protected RepairQueueNode prev = null;
    // a pointer onto the next node
    protected RepairQueueNode next = null;
    // the node payload
    protected RepairDigramRecord payload = null;

    /**
     * Constructor.
     * 
     * @param digramRecord the payload to wrap.
     */
    public RepairQueueNode(RepairDigramRecord digramRecord) {
      this.payload = digramRecord;
    }

    /**
     * The occurrence frequency getter.
     * 
     * @return the digram occurrence frequency.
     */
    public int getFrequency() {
      return this.payload.freq;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + getOuterType().hashCode();
      result = prime * result + ((next == null) ? 0 : next.hashCode());
      result = prime * result + ((payload == null) ? 0 : payload.hashCode());
      result = prime * result + ((prev == null) ? 0 : prev.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      RepairQueueNode other = (RepairQueueNode) obj;
      if (!getOuterType().equals(other.getOuterType()))
        return false;
      if (next == null) {
        if (other.next != null)
          return false;
      }
      else if (!next.equals(other.next))
        return false;
      if (payload == null) {
        if (other.payload != null)
          return false;
      }
      else if (!payload.equals(other.payload))
        return false;
      if (prev == null) {
        if (other.prev != null)
          return false;
      }
      else if (!prev.equals(other.prev))
        return false;
      return true;
    }

    private RepairPriorityQueue getOuterType() {
      return RepairPriorityQueue.this;
    }

  }

  // public void runCheck() {
  //
  // HashSet<String> keys = new HashSet<String>();
  // for (String s : this.elements.keySet()) {
  // keys.add(s);
  // }
  //
  // RepairQueueNode hp = this.head;
  // while (null != hp) {
  // String str = hp.payload.str;
  // keys.remove(str);
  // hp = hp.next;
  // }
  // if (!(keys.isEmpty())) {
  // System.out.println(keys);
  // throw new RuntimeException("tracking arror here");
  // }
  //
  // }

}
