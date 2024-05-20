import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TFIDF {
    private static final String TOTAL_DOC = "TotalDoc";
    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");
            String docId = parts[0];
            String freq = parts[1];

            context.write(
                    new Text(docId),
                    new Text(key.toString() + " " + freq)
            );
        }
    }

    public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
        static long totalDoc = 0;
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            // total terms in a doc
            int count = 0;
            Map<String, Integer> aggKeys = new HashMap<>();
            for (Text value : values) {
                count++;
                String[] parts = value.toString().split("\\s+");
                //              termId              docId
                String aggKey = parts[0] + " " + key.toString();
                aggKeys.put(aggKey, Integer.parseInt(parts[1]));
            }

            int finalCount = count;
            for (Map.Entry<String, Integer> entry : aggKeys.entrySet()) {
                float tf = (float) entry.getValue() / finalCount;
                context.write(
                        new Text(entry.getKey()),
                        new Text(Float.toString(tf))
                );
            }

            totalDoc++;
        }

        @Override
        protected void cleanup(Context context) {
            context.getCounter(TOTAL_DOC, TOTAL_DOC).increment(totalDoc);
        }
    }

    public static class TokenizerMapperJ2 extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");
            String docId = parts[0];
            String freq = parts[1];

            context.write(
                    new Text(key.toString()), // term as key
                    new Text(docId + " " + freq)
            );
        }
    }

    public static class IntSumReducerJ2 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Configuration config = context.getConfiguration();
            int totalDoc = Integer.parseInt(config.get(TOTAL_DOC));

            // total terms in a doc
            int count = 0;
            HashSet<String> aggKeys = new HashSet<>();
            for (Text value : values) {
                count++;
                String[] parts = value.toString().split("\\s+");
                //              termId              docId
                aggKeys.add(key.toString() + " " + parts[0]);
            }


            int finalCount = count;
            for (String aggKey : aggKeys) {
                double idf = Math.log((double) totalDoc / finalCount);
                context.write(
                        new Text(aggKey),
                        new Text(Double.toString(idf))
                );
            }
        }
    }

    public static class FinalMapper extends Mapper<Object, Text, Text, FloatWritable> {
        Map<String, Float> TF = null;
        Map<String, Float> IDF = null;
        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);

            TF  = FileReader(fs, conf.get("TFPath"));
            IDF = FileReader(fs, conf.get("IDFPath"));
        }

        private Map<String, Float> FileReader(FileSystem fs, String tfPath) throws IOException {
            Map<String, Float> map = new HashMap<>();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(tfPath))))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\\s+");

                    map.put(parts[0] + "\t" + parts[1], Float.parseFloat(parts[2]));
                }
            }

            return map;
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");

            String aggKey = key.toString() + "\t" + parts[0];
            float tfidf = TF.get(aggKey) * IDF.get(aggKey);

            context.write(new Text(aggKey), new FloatWritable(tfidf));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("TFPath", args[1] + ".tf.tmp/part-r-00000");
        conf.set("IDFPath", args[1] + ".idf.tmp/part-r-00000");

        Job job = Job.getInstance(conf, "tf");
        job.setJarByClass(TFIDF.class);

        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(IntSumReducer.class);

        job.setInputFormatClass(KeyValueTextInputFormat.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1] + ".tf.tmp"));

        job.waitForCompletion(true);

        // ------------------------
        long totalDoc = job.getCounters().findCounter(TOTAL_DOC, TOTAL_DOC).getValue();

        conf.set(TOTAL_DOC, Long.toString(totalDoc));

        Job job2 = Job.getInstance(conf, "tdf");
        job2.setJarByClass(TFIDF.class);

        job2.setMapperClass(TokenizerMapperJ2.class);
        job2.setReducerClass(IntSumReducerJ2.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        job2.setInputFormatClass(KeyValueTextInputFormat.class);

        FileInputFormat.addInputPath(job2, new Path(args[0]));
        FileOutputFormat.setOutputPath(job2, new Path(args[1] + ".idf.tmp"));

        job2.waitForCompletion(true);
        // ------------------------
        Job job3 = Job.getInstance(conf, "tdf");
        job3.setJarByClass(TFIDF.class);

        job3.setMapperClass(FinalMapper.class);

        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(FloatWritable.class);

        job3.setInputFormatClass(KeyValueTextInputFormat.class);

        FileInputFormat.addInputPath(job3, new Path(args[0]));
        FileOutputFormat.setOutputPath(job3, new Path(args[1]));

        System.exit(job3.waitForCompletion(true) ? 0 : 1);

    }
}