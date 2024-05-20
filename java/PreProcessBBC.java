import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PreProcessBBC {

    public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private HashSet<String> stopWords = new HashSet<>();
        private HashMap<String, Integer> termIds = new HashMap<>();
        private HashMap<String, Integer> documentIds = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            String basePath = conf.get("basePath");

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(basePath + "/_ignored/stopwords.txt"))))) {
                String stopWord;
                while ((stopWord = reader.readLine()) != null) {
                    stopWords.add(stopWord);
                }
            }

            int counter = 1;
            // Reading bbc.terms
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(basePath + "/_ignored/bbc.terms"))))) {
                String term;
                while ((term = reader.readLine()) != null) {
                    termIds.put(term, counter);
                    counter += 1;
                }
            }

            counter = 1;
            // Reading bbc.docs
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(basePath + "/_ignored/bbc.docs"))))) {
                String document;
                while ((document = reader.readLine()) != null) {
                    documentIds.put(document, counter);
                    counter += 1;
                }
            }
        }

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            Path filePath = fileSplit.getPath();

            String doc = filePath.getParent().getName();
            String name = filePath.getName().split("\\.")[0];

            // skip file bbc.docs, bbc.terms, stopwords.txt because we put these file
            // in the same directory of input file.
            if (doc.equals("_ignore")) { return; }

            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String term = itr.nextToken()
                        .replaceAll("/[^A-Z0-9£]/ig", "")
                        .toLowerCase();

                if (stopWords.contains(term)) { continue; }

                if (!termIds.containsKey(term)) { continue; }
                
                word.set(String.format("%d %d", termIds.get(term), documentIds.get(doc+"."+name))) ;
                context.write(word, one);
            }
            
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            IntWritable result = new IntWritable();
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("basePath", args[0]);

        Job job = Job.getInstance(conf, "Tokenizer");
        job.setJarByClass(PreProcessBBC.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileInputFormat.setInputDirRecursive(job, true);

        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
