package net.seninp.tinker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.discord.DiscordRecord;
import net.seninp.jmotif.sax.discord.DiscordRecords;
import net.seninp.jmotif.sax.discord.HOTSAXImplementation;

public class PaperDiscordFinderHOTSAX {

  private static final String IN_DATA = "RCode/TKDD/sine_5anomalies_rwalk_04noise.txt";

  private static final String IN_PARAMS = "RCode/TKDD/sine_5anomalies_rwalk_04noise_sampler_out.txt";

  private static final Logger LOGGER = LoggerFactory.getLogger(PaperDiscordFinderHOTSAX.class);

  private static final String COMMA = ",";

  private static final String CR = "\n";

  public static void main(String[] args) throws Exception {

    double[] ts = TSProcessor.readFileColumn(IN_DATA, 0, 0);
    LOGGER.info("read " + ts.length + " points from " + IN_DATA);

    BufferedWriter bw = new BufferedWriter(
        new FileWriter(new File("RCode/TKDD/sine_5anomalies_hotsax_rwalk_04noise_discord_res_10.txt")));

    BufferedReader br = new BufferedReader(new FileReader(new File(IN_PARAMS)));

    String line = null;
    while ((line = br.readLine()) != null) {

      String[] split = line.trim().split(",");
      Integer WIN = Integer.valueOf(split[2]);
      Integer PAA = 3;
      Integer ALP = 3;

      StringBuilder logStr = new StringBuilder();

      logStr.append(WIN).append(COMMA);
      logStr.append(PAA).append(COMMA);
      logStr.append(ALP).append(COMMA);

      //
      // ************************
      //

      // run HOTSAX with this intervals set
      //
      DiscordRecords discords = HOTSAXImplementation.series2Discords(ts, 10, WIN, PAA, ALP,
          NumerosityReductionStrategy.NONE, 0.01);
      //
      // ************************
      //
      //
      // let's see the discords...

      int[] discordsFound = { 0, 0, 0, 0, 0 };
      ArrayList<Interval> discordIntervals = new ArrayList<Interval>();
      // discord #1
      Interval i = new Interval(205, 217);
      discordIntervals.add(i);
      // discord #2
      i = new Interval(360, 392);
      discordIntervals.add(i);
      // discord #3
      i = new Interval(726, 740);
      discordIntervals.add(i);
      // discord #4
      i = new Interval(500, 545);
      discordIntervals.add(i);
      // discord #5
      i = new Interval(1081, 1095);
      discordIntervals.add(i);

      for (DiscordRecord discord : discords) {

        int pos = discord.getPosition();
        int len = discord.getLength();
        
        System.out.println(len);

        Interval di = new Interval(pos, pos + len);

        for (int intIdx = 0; intIdx < discordIntervals.size(); intIdx++) {
          Interval interval = discordIntervals.get(intIdx);
          if (interval.intersects(di)) {
            discordsFound[intIdx] = discordsFound[intIdx] + 1;
          }
        }

      }

      for (int j = 0; j < discordsFound.length; j++) {
        logStr.append(discordsFound[j]).append(",");
      }

      logStr.delete(logStr.length() - 1, logStr.length());

      bw.write(logStr.toString() + CR);

    }
    br.close();

    bw.close();
  }

}
