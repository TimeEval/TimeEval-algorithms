package net.seninp.tinker;

public class PagePrinter {

  public static void main(String[] args) {

    for (int i = 1; i < 101; i++) {
      System.out
          .println("<div style=\"display:block;text-align:left\">"+"<a href=\"https://sites.google.com/a/seninp.net/anomaly/a3/A3Benchmark-TS"
              + i
              + ".csv.png?attredirects=0\" imageanchor=\"1\">"
              + "<img border=\"0\" src=\"https://sites.google.com/a/seninp.net/anomaly/a3/A3Benchmark-TS"
              + i
              + ".csv.png\">"
              + "</a></div>"
              + "<div style=\"display:block;text-align:left\"><font color=\"#0000ff\">*******************</font></div><br>");
    }

  }
}
