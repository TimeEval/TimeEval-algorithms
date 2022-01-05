package net.seninp.grammarviz.cli;

import net.seninp.gi.logic.GrammarRuleRecord;
import net.seninp.gi.logic.GrammarRules;
import net.seninp.gi.logic.RuleInterval;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Write GrammarRules to an output file.
 */
public class RulesWriter {
    private static final Logger logger = LoggerFactory.getLogger(RulesWriter.class);
    private static final String NEWLINE = System.lineSeparator();

    private String fname;
    private BufferedWriter writer;

    public RulesWriter(String fname) {
        this.fname = fname;
        this.writer = null;
    }

    public RulesWriter write(GrammarRules rules) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(this.fname)))) {
            this.writer = writer;
            this.writeHeader()
                .writeRules(rules);
        } catch (IOException e) {
            if (logger.isDebugEnabled()) {
                logger.error("Encountered error while writing stats file: " + e.getMessage(), e);
            } else {
                logger.error("Encountered error while writing stats file: " + e.getMessage());
            }
        }
        return this;
    }

    private RulesWriter writeHeader() throws IOException {
        String output = "# filename: " + fname + NEWLINE +
                "# sliding window: " + TS2GrammarParameters.SAX_WINDOW_SIZE + NEWLINE +
                "# paa size: " + TS2GrammarParameters.SAX_PAA_SIZE + NEWLINE +
                "# alphabet size: " + TS2GrammarParameters.SAX_ALPHABET_SIZE + NEWLINE;
        writer.write(output);
        return this;
    }

    private void writeRules(GrammarRules rules) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (GrammarRuleRecord ruleRecord : rules) {
            sb.setLength(0);  // reset

            sb.append("/// ").append(ruleRecord.getRuleName()).append(NEWLINE);
            sb.append(ruleRecord.getRuleName()).append(" -> \'")
              .append(ruleRecord.getRuleString().trim()).append("\', expanded rule string: \'")
              .append(ruleRecord.getExpandedRuleString()).append("\'").append(NEWLINE);

            if (!ruleRecord.getOccurrences().isEmpty()) {
                List<RuleInterval> intervals = ruleRecord.getRuleIntervals();
                int[] starts = new int[intervals.size()];
                int[] lengths = new int[intervals.size()];
                for (int i = 0; i < intervals.size(); i++) {
                    starts[i] = intervals.get(i).getStart();
                    lengths[i] = intervals.get(i).getEnd() - intervals.get(i).getStart();
                }

                sb.append("subsequences starts: ").append(Arrays.toString(starts)).append(NEWLINE);
                sb.append("subsequences lengths: ").append(Arrays.toString(lengths)).append(NEWLINE);
            }

            sb.append("rule occurrence frequency ").append(ruleRecord.getOccurrences().size()).append(NEWLINE);
            sb.append("rule use frequency ").append(ruleRecord.getRuleUseFrequency()).append(NEWLINE);
            sb.append("min length ").append(ruleRecord.minMaxLengthAsString().split(" - ")[0]).append(NEWLINE);
            sb.append("max length ").append(ruleRecord.minMaxLengthAsString().split(" - ")[1]).append(NEWLINE);
            sb.append("mean length ").append(ruleRecord.getMeanLength()).append(NEWLINE);

            writer.write(sb.toString());
        }
    }
}
