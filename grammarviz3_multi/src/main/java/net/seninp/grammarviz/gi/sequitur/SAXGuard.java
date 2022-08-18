package net.seninp.grammarviz.gi.sequitur;

/*
 This class is part of a Java port of Craig Nevill-Manning's Sequitur algorithm.
 Copyright (C) 1997 Eibe Frank

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

import net.seninp.grammarviz.gi.sequitur.SAXRule;

/**
 * 
 * The Guard node - serves as head of the "doubly-linked" list representing a rule - in other words
 * it's a rule handler. Adaption of Eibe Frank code for JMotif API.
 * 
 * 
 * @author Manfred Lerner, seninp
 * 
 */
public class SAXGuard extends SAXSymbol {

  // To ensure that rules can be lengthened and shortened efficiently, SEQUITUR represents a rule
  // using a doubly-linked list whose start and end are connected to a single guard node, shown for
  // two rules A and B in Figure 3. The guard node also serves as an attachment point for the
  // left-hand side of the rule, because it remains constant even when the rule contents change.
  // Each nonterminal symbol also points to the rule it heads, shown in Figure 3 by the pointer from
  // the nonterminal symbol B in rule A to the head of rule B. With these pointers, no arrays are
  // necessary for accessing rules or symbols, because operations only affect adjacent symbols or
  // rules headed by a non-terminal.

  protected SAXRule r;

  /**
   * Constructor.
   * 
   * @param theRule The guard holds.
   */
  public SAXGuard(SAXRule theRule) {
    r = theRule;
    value = null;
    p = this;
    n = this;
  }

  /**
   * {@inheritDoc}
   */
  public void cleanUp() {
    System.err.println("Attempting to cleaning up the Guard");
    join(p, n);
  }

  /**
   * {@inheritDoc}
   */
  public boolean isGuard() {
    return true;
  }

  /**
   * {@inheritDoc}
   */
  public void deleteDigram() {
    // Does nothing
  }

  /**
   * {@inheritDoc}
   */
  public boolean check() {
    return false;
  }
}
