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

import net.seninp.grammarviz.gi.sequitur.SAXSymbol;

/**
 * This is a regular, a.k.a. "terminal", symbol. Adaption of Eibe Frank code for JMotif API.
 * 
 * 
 * @author Manfred Lerner
 * 
 */
public class SAXTerminal extends SAXSymbol implements Cloneable {

  /**
   * Constructor.
   * 
   * @param symbol the symbol's value.
   * @param originalPosition this symbol original position in the string.
   */
  public SAXTerminal(String symbol, int originalPosition) {
    this.value = symbol;
    this.originalPosition = originalPosition;
    p = null;
    n = null;
  }

  /**
   * {@inheritDoc}
   */
  public void cleanUp() {
    join(p, n);
    deleteDigram();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public String toString() {
    return "SAXterminal [value=" + value + ", p=" + p + ", n=" + n + ", position: "
        + originalPosition + "]";
  }
}
